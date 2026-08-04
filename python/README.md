# CTKM Extractor — OCR trích xuất chương trình khuyến mại (CTKM) → JSON

Đọc **1 ảnh** chứa bảng mô tả chương trình khuyến mại viễn thông và trích xuất ra
JSON:

```json
{
  "packageCode": "CTKMN180X",
  "registerFee": 163636.3636,
  "monthlyFee": 0,
  "onnetMinutes": "MP 20p đầu tiên",
  "offnetMinutes": 150,
  "sms": 100,
  "dataGB": 60,
  "youtubeGB": 25,
  "spotifyGB": 25,
  "cycle": "tháng",
  "allowedPackages": ["Basic+", "Family", "Corporate++"]
}
```

Nguyên tắc thiết kế: **không hard-code dữ liệu theo ảnh mẫu**. Toàn bộ tri thức về
mẫu bảng (tên header, từ đồng nghĩa, parser, regex fallback) nằm trong
`ctkm_extractor/extraction/schema.yaml`; thêm field mới hoặc hỗ trợ mẫu CTKM khác
chỉ cần sửa file YAML này.

Bản Python này **độc lập hoàn toàn** với bản C++ (`../cpp/`): cài đặt và chạy được
mà không cần bản kia. Hai bản dùng cùng thuật toán và cho cùng kết quả JSON. Mọi
lệnh dưới đây chạy từ thư mục `python/`.

---

## 1. Kiến trúc 5 tầng

```
ctkm_extractor/
  ocr/                          # Tầng 1: ảnh -> token (text + bbox + confidence)
    base.py                     #   OCRProvider, OCRToken, OCRResult, BoundingBox
    preprocess.py               #   grayscale -> deskew -> adaptive threshold (OpenCV)
    paddle_vietocr_provider.py  #   MẶC ĐỊNH: PP-OCRv5 detect (DB) + VietOCR recognize
    tesseract_ocr.py            #   FALLBACK: Tesseract (nhẹ, không cần tải model)
  table/                        # Tầng 2: token -> cấu trúc bảng
    morphology.py                #   ƯU TIÊN NHẤT: dò đường kẻ bảng bằng CV cổ điển
    pp_structure.py             #   FALLBACK 1: PP-Structure table recognition (ô gộp)
    reconstruct.py              #   FALLBACK 2: cluster bounding box theo y rồi x
    __init__.py                 #   build_table(): điều phối 4 mức fallback
  extraction/                   # Tầng 3: bảng/text -> field có kiểu
    schema.yaml                 #   khai báo field: aliases + parser + regex fallback
    field_parsers.py            #   registry parser (money/int/gb/cycle/list/text)
    extractor.py                #   orchestrator: OCR -> table -> map -> parse -> JSON
  cli.py                        # Tầng 4: giao diện dòng lệnh
  tests/                        # pytest cho từng tầng
```

Mỗi tầng test được độc lập: tầng OCR nhận ảnh, tầng table nhận token, tầng
extraction nhận bảng **hoặc** raw text, CLI nhận đường dẫn.

### Tầng 1 — OCR: tách detection và recognition

| Bước | Model | Lý do |
| --- | --- | --- |
| Detection | `paddleocr` PP-OCRv5, thuật toán **DB** (Differentiable Binarization), chạy ở chế độ chỉ detect | PP-OCRv5 chịu nhiễu/watermark tốt và cho bbox dòng/từ chính xác hơn engine truyền thống |
| Recognition | **VietOCR** `vgg_transformer` (pretrained trên corpus tiếng Việt) | Recognizer đa ngôn ngữ mặc định của PaddleOCR sai dấu tiếng Việt rõ rệt; Tesseract còn hạn chế hơn nữa với dấu và tài liệu nhiễu |

Pipeline: tiền xử lý (grayscale → deskew → adaptive threshold) → detect polygon →
crop + nắn phối cảnh từng vùng → VietOCR nhận dạng → `OCRResult` gồm danh sách
token `text + bbox + confidence`.

Provider `tesseract` giữ **nguyên interface** `OCRProvider.extract(image_path) ->
OCRResult`, dùng khi môi trường không cài được PaddleOCR/VietOCR. Việc import
`paddleocr`/`vietocr`/`torch` được bọc `try/except ImportError` **tại thời điểm
khởi tạo provider**, thiếu thì tự fallback kèm log warning — chương trình không
bao giờ crash ngay từ import.

### Tầng 2 — Table: 4 mức fallback

1. **Morphology** (`table/morphology.py`, ưu tiên nhất): tách đường kẻ ngang/dọc
   bằng morphological opening (OpenCV), suy ra lưới hàng từ đường kẻ ngang đã dò
   và cột từ **đường kẻ dọc thật** của dải hàng đang xét. Bảng không có đủ kẻ dọc
   thì cột mới suy từ toạ độ token **của riêng hàng header** (đã kiểm chứng trên
   ảnh scan thật: gộp toạ độ mọi hàng làm 2 cột hẹp cạnh nhau bị nối nhầm do
   token hàng dữ liệu tràn ra ngoài cột). Band suy từ header chỉ rộng bằng chữ
   trong header, nên token dữ liệu dài hơn tiêu đề cột có thể **không chồng lấn
   band nào**; token không chồng lấn band nào — ở cả hai nhánh — được gán vào cột
   **gần nhất theo khoảng cách** thay vì rơi về cột 0 và dính vào ô nhãn. Token nằm dưới đường kẻ cuối trong phạm vi một bước hàng vẫn được giữ
   như **hàng cuối chưa đóng viền** (ảnh scan hay bị cắt mất đường kẻ dưới), còn
   chữ ở chân trang cách bảng quá xa thì bị loại. Tất định, không cần tải model,
   không phụ thuộc `paddleocr` nên chạy được cả khi dùng provider fallback
   `tesseract`. Yêu cầu tối thiểu 3 đường kẻ ngang; bảng không viền hoặc dò được
   quá ít đường kẻ sẽ raise và rơi xuống mức 2.
2. **PP-Structure** (`PPStructure(table=True, ocr=False)`): nhận trực tiếp
   hàng/cột/ô gộp bằng model học sâu — dự phòng cho bảng không viền rõ mà (1)
   không dò được đường kẻ. Text **không** lấy từ recognizer của Paddle mà lấy từ
   VietOCR ở tầng 1, ghép vào ô theo diện tích chồng lấn.
3. **Cluster bounding box** thủ công: gom token thành hàng theo trục y, thành cột
   theo hình chiếu trên trục x.
4. **Raw text blob**: coi toàn bộ text là một khối, để tầng 3 dò bằng regex.

### Tầng 3 — Extraction: schema-driven

Với mỗi field, thứ tự thử (dừng ở kết quả parse được đầu tiên):

1. Khớp **alias** trong bảng. Nếu nhãn nằm ở **hàng header** của bảng nhiều cột
   (`n_cols > 1`), ưu tiên thử ô **bên dưới/phải** trước phần dư ngay trong ô
   nhãn — tránh trường hợp chú thích ngắn dính kèm trong ô header (VD "(TK533)")
   bị nhầm là giá trị thay vì ô dữ liệu thật bên dưới. Với nhãn không ở hàng
   header (bảng nhãn-giá trị 1 cột, hoặc dòng dạng `"Chu kỳ gia hạn: tháng"`),
   thử phần dư cùng ô trước như cũ. Ứng viên trông giống *nhãn* của field khác bị
   đẩy xuống cuối danh sách.
2. Khớp alias theo **từng dòng** raw text.
3. **Regex fallback** khai báo trong `schema.yaml`.
4. Không tìm được → ghi log warning và trả `null` (không raise).

Sau khi mọi field đã resolve, một bước hậu kiểm phát hiện **2 field khác nhau
vô tình lấy giá trị từ cùng một ô bảng** (VD alias fuzzy-match khớp nhầm cột) —
chỉ giữ field có score cao hơn, field còn lại trả `null` kèm cảnh báo thay vì
âm thầm trả cùng một giá trị sai cho cả hai. Field khai báo `keyword` trong
`parser_args` (VD `youtubeGB`/`spotifyGB` cùng đọc từ 1 ô gộp theo sub-label
riêng) được loại trừ khỏi kiểm tra này vì đó là thiết kế có chủ đích.

---

## 2. Cài đặt

### 2.1. Provider mặc định (PaddleOCR + VietOCR)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install paddlepaddle paddleocr vietocr
```

* Bản **CPU** là đủ, **không bắt buộc GPU** (`paddlepaddle` bản thường đã chạy được).
* Hỗ trợ **cả `paddleocr` 2.x lẫn 3.x**. Provider tự dò API khả dụng:
  * `paddleocr` >= 3.0: dùng class `TextDetection` (chỉ nạp model DB, **không** tải
    kèm model recognition ~77 MB vốn không được dùng), gọi qua `predict()`.
    Lưu ý: ở 3.x `ocr()` chỉ còn là alias deprecated của `predict()` và **không**
    nhận `det`/`rec` nữa.
  * `paddleocr` 2.x: dùng `PaddleOCR(...).ocr(img, det=True, rec=False, cls=False)`.
* **oneDNN/mkldnn mặc định TẮT** trên CPU. Tổ hợp `paddlepaddle` mới (IR = PIR) +
  oneDNN có thể chết ngay lúc inference với
  `(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
  [pir::ArrayAttribute<pir::DoubleAttribute>] (at .../onednn_instruction.cc)`.
  Muốn chạy nhanh hơn thì bật lại; nếu backend lỗi, provider tự tắt oneDNN và thử
  lại một lần:

  ```python
  from ctkm_extractor.extraction.extractor import CTKMExtractor
  CTKMExtractor(engine="paddle_vietocr", provider_kwargs={"enable_mkldnn": True})
  ```
* **PaddleX chỉ khởi tạo được MỘT lần cho mỗi process** (`PDX has already been
  initialized. Reinitialization is not supported.`). Vì vậy detector được cache
  theo process và provider thứ hai dùng lại detector đã dựng, kể cả khi tham số
  khác đi — dựng mới là bất khả thi chứ không phải lựa chọn. Muốn thật sự đổi
  tham số detector thì phải chạy tiến trình mới (hoặc restart kernel notebook).
* **Lần chạy đầu tiên sẽ tự tải model về cache local**: DB text detector của
  PP-OCRv5, model PP-Structure table, và weights VietOCR `vgg_transformer`. Vì vậy
  **cần internet đúng 1 lần**; các lần chạy sau đọc từ cache và hoạt động offline
  bình thường (cache của PaddleOCR thường nằm ở `~/.paddleocr/` hoặc `~/.paddlex/`,
  weights VietOCR nằm trong thư mục cache do `vietocr` cấu hình).

### 2.2. Provider fallback (Tesseract)

Chỉ cần khi không cài được PaddleOCR/VietOCR.

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-vie

# macOS (Homebrew)
brew install tesseract tesseract-lang

# Windows: cài bằng installer của UB Mannheim rồi thêm thư mục cài đặt vào PATH
#   https://github.com/UB-Mannheim/tesseract/wiki
#   (nhớ tick "Vietnamese" trong phần Additional language data)
```

Kiểm tra gói tiếng Việt: `tesseract --list-langs | grep vie`.

### 2.3. Toàn bộ dependency

```bash
pip install -r requirements.txt
```

### 2.4. PDF nhiều trang

Bảng CTKM thường **chỉ nằm ở một trang** trong cả tập hồ sơ, nên pipeline không
giả định trang nào chứa bảng:

1. Render **từng trang** ra PNG 300 DPI (`pymupdf`), OCR độc lập từng trang.
2. Chấm điểm mỗi trang = *(số field trích được, điểm khớp trung bình)*. Trang
   không có bảng ra `(0, 0.0)` nên không bao giờ được chọn khi có trang khác.
3. **Trang chính** = trang điểm cao nhất; hoà thì lấy trang có số nhỏ hơn.
4. Field nào **vẫn thiếu** mới lấy bù từ các trang còn lại theo thứ tự điểm giảm
   dần — cho trường hợp bảng bị tách qua nhiều trang. Bước này **không ghi đè**
   giá trị của trang chính: trang chính là trang thật sự chứa bảng, còn trang
   khác dễ có nhãn trùng tên nằm trong phần văn bản thường.

Mỗi lần lấy bù đều ghi rõ trong `warnings`, và `--debug` in ra trang nguồn của
từng field:

```
Trang            : 2 (chính) trong 3 trang đã OCR
cycle            = 'tháng'   [source=text_line, score=0.88, raw='tháng', trang=3]
```

```bash
pip install pymupdf                       # chỉ cần khi đầu vào là PDF
python -m ctkm_extractor.cli --pdf ho_so.pdf --out result.json --debug
python -m ctkm_extractor.cli --pdf ho_so.pdf --pages 2-3   # biết trước trang nào
```

Thiếu `pymupdf` thì CLI báo lỗi kèm hướng dẫn cài, không traceback.

### 2.5. Cài như một package (tuỳ chọn)

```bash
cd python
pip install .              # chỉ dependency nhẹ: PyYAML, OpenCV, Pillow, pytesseract
pip install ".[paddle]"    # thêm provider mặc định (paddlepaddle + paddleocr + vietocr + torch)
pip install ".[test]"      # thêm pytest
```

Sau khi cài, chạy được từ bất kỳ thư mục nào bằng lệnh `ctkm-extractor` (schema đi
kèm trong package nên không cần truyền `--schema`):

```bash
ctkm-extractor --image ../sample.png --out result.json
```

---

## 3. Chạy chương trình

```bash
# Mặc định: PP-OCRv5 detect + VietOCR recognize + morphology (đường kẻ bảng)
python -m ctkm_extractor.cli --image ../sample.png --out result.json

# In JSON ra stdout
python -m ctkm_extractor.cli --image ../sample.png

# Ép dùng provider cụ thể
python -m ctkm_extractor.cli --image ../sample.png --out result.json --engine paddle_vietocr
python -m ctkm_extractor.cli --image ../sample.png --out result.json --engine tesseract

# Debug: in OCR raw text, bảng đã dựng và nguồn của từng field ra stderr
python -m ctkm_extractor.cli --image ../sample.png --out result.json --debug

# Xem engine nào khả dụng trong môi trường hiện tại
python -m ctkm_extractor.cli --list-engines
```

Các cờ khác:

| Cờ | Ý nghĩa |
| --- | --- |
| `--schema <path>` | Dùng `schema.yaml` tuỳ biến thay cho schema đi kèm package |
| `--no-morphology` | Bỏ qua dò bảng bằng đường kẻ (CV cổ điển), dùng thẳng PP-Structure |
| `--no-pp-structure` | Bỏ qua PP-Structure, dùng thẳng fallback cluster bounding box |
| `--strict-engine` | Không tự fallback sang engine khác khi engine yêu cầu thiếu dependency |
| `--pdf <path>` | PDF **nhiều trang**: tách từng trang, OCR từng trang rồi gộp kết quả. `--image` trỏ vào `.pdf` cũng chạy đường này |
| `--pages <spec>` | Chỉ xử lý một số trang, VD `"1,3-5"`; bỏ trống = mọi trang |
| `--dpi <n>` | Độ phân giải render mỗi trang (mặc định 300 - mức đã kiểm chứng cả pipeline) |
| `--keep-pages` | Giữ lại ảnh từng trang đã render để soi khi debug |
| `--image <path>` | Một ảnh, hoặc một file `.pdf`. Nhiều trang thì dùng `--pdf` (bản C++ mới là bản nhận `--image` lặp lại) |
| `--no-binarize` / `--binarize` | Ép TẮT/BẬT adaptive threshold. **Mặc định chương trình chạy cả hai rồi giữ kết quả trích được nhiều field hơn**; ép để chỉ chạy một lượt (nhanh gấp đôi) |
| `--text-file <path>` | Bỏ qua OCR, đọc raw text từ file — tiện debug riêng tầng trích xuất |
| `--indent <n>` | Số space thụt lề JSON (mặc định 2) |

CLI luôn ghi JSON **đủ field** (thiếu thì `null`) kể cả khi OCR lỗi; exit code khác
0 chỉ khi lỗi cấu hình (schema hỏng, không tìm thấy ảnh, không có engine nào).

#### Khi nào nên ép `--binarize` / `--no-binarize`

Mặc định chương trình chạy **cả hai** cấu hình rồi giữ lần trích được nhiều field
hơn, nên thường không cần đụng tới hai cờ này. Ép một cấu hình có hai lý do:

* **Nhanh gấp đôi** khi xử lý hàng loạt ảnh cùng loại đã biết trước.
* **Tránh chọn nhầm**: hàm chấm điểm đếm số field *có giá trị*, không phân biệt
  được giá trị đúng với giá trị rác — xem [ghi chú và số liệu đo ở README
  gốc](../README.md#chọn-tiền-xử-lý-tự-động-và-giới-hạn-của-nó).

Kinh nghiệm đo trên biểu mẫu BM.12 thật (watermark chéo đè lên chữ): với engine
`tesseract`, `--no-binarize` thường đúng hơn, vì adaptive threshold biến nét
watermark mờ thành nét đen đặc đè lên chữ. Với `paddle_vietocr` thì ngược lại —
detector DB huấn luyện trên ảnh tự nhiên, bỏ nhị phân hoá chỉ làm nó bắt thêm
token rác và sinh giá trị sai.

Dùng như thư viện:

```python
from ctkm_extractor.extraction.extractor import CTKMExtractor

extractor = CTKMExtractor(engine="paddle_vietocr")
result = extractor.extract_from_image("../sample.png")
print(result.to_json())
print(result.debug_report())      # raw text + bảng + nguồn từng field
```

---

## 4. Chạy test

```bash
pip install pytest PyYAML
python -m pytest ctkm_extractor/tests -q
```

Bộ test **không cần ảnh thật, không cần paddleocr/vietocr**: OCR được mock bằng
raw-text blob.

* `tests/test_field_parsers.py` — từng parser với input giả lập, gồm cả chuỗi rỗng,
  thiếu field, sai ký tự do OCR (`"l63,636.3636"`, `"15O.OOO"`, `"1OO"`), tham số
  sai kiểu; chứng minh không crash và trả `null` hợp lý.
* `tests/test_extractor.py` — orchestrator end-to-end với fixture blob (assert đúng
  cấu trúc JSON và giá trị), mẫu CTKM khác nhãn/khác thứ tự, ảnh nhiễu, OCR ném
  exception, bảng dọc/bảng ngang/đổi thứ tự cột, 3 mức fallback của tầng table, và
  CLI ghi file JSON.

---

## 5. Thêm field mới hoặc mẫu CTKM mới — chỉ sửa `schema.yaml`

Đây là điểm nhấn thiết kế: **không sửa code Python**.

### Thêm một field mới

```yaml
  - name: tiktokGB          # tên khoá trong JSON output
    type: number            # number | string | array
    parser: gb_parser       # tên hàm trong field_parsers.py
    parser_args:
      keyword: "Tiktok"     # tham số truyền cho parser
      fallback_to_number: true
    aliases:                # mọi cách viết header/nhãn có thể gặp
      - "Tiktok"
      - "Data Tiktok"
      - "Ưu đãi Tiktok"
    regex:                  # fallback khi không dựng được bảng
      - "(tiktok[^0-9\\n]{0,30}[0-9][0-9.,]*\\s*g\\s?b)"
```

### Hỗ trợ một mẫu CTKM khác

Mẫu khác thường chỉ khác **tên cột / thứ tự cột**: thêm cách viết đó vào `aliases`
của field tương ứng. Thứ tự cột không quan trọng — extractor dò theo nhãn chứ không
theo chỉ số cột. Nếu mẫu mới có bố cục lạ, dùng `--schema schema_mau_moi.yaml` để
giữ song song nhiều schema.

### Ghi chú về `regex`

Pattern được so khớp trên bản text **đã bỏ dấu** (phép bỏ dấu giữ nguyên độ dài
chuỗi), nên hãy viết pattern không dấu: `cuoc\s*dang\s*ky`. Giá trị trả về vẫn được
cắt từ text gốc **còn dấu**, nên `cycle` vẫn ra `"tháng"` chứ không phải `"thang"`.

### Parser có sẵn

| Parser | Chức năng | Ví dụ |
| --- | --- | --- |
| `money_parser` | Số tiền lẫn lộn `,` và `.` | `"163,636.3636"` → `163636.3636`; `"150.534.213"` → `150534213` |
| `int_parser` | Số nguyên đầu tiên | `"MP 20p đầu tiên"` → `20`; `"1.500 phút"` → `1500` |
| `gb_parser` | Dung lượng GB đầu tiên | `"60GB/tháng, tối đa 8gb/1 ngày"` → `60` |
| `cycle_parser` | Từ khoá chu kỳ | `"Chu kỳ gia hạn: tháng"` → `"tháng"` |
| `list_parser` | Tách danh sách theo dấu phẩy | `"Basic+, Family, Corporate++"` → `["Basic+", "Family", "Corporate++"]` |
| `text_parser` | Chuỗi đã làm sạch, lọc theo pattern | `" : ctkmn180x "` → `"CTKMN180X"` |

Cần logic mới thì viết thêm một hàm trong `field_parsers.py` với decorator
`@register("ten_parser")` — orchestrator tự tra registry theo tên trong YAML.

#### Heuristic của `money_parser`

Dấu phân cách trong CTKM rất lộn xộn nên quy tắc là:

1. Có **cả** `,` và `.` → dấu **xuất hiện sau cùng** là dấu thập phân
   (`"163,636.3636"` → `163636.3636`, `"1.234,56"` → `1234.56`).
2. Chỉ **một loại** dấu, xuất hiện nhiều lần và mọi nhóm phía sau đều đúng 3 chữ số
   → tất cả là phân cách hàng nghìn (`"150.534.213"` → `150534213`). Nếu áp dụng
   máy móc quy tắc "dấu cuối cùng luôn là thập phân" thì trường hợp này sẽ ra
   `150534.213` — sai.
3. Chỉ một dấu duy nhất → hàng nghìn nếu nhóm sau đúng 3 chữ số, ngược lại là thập
   phân (`"150.000"` → `150000`, `"163,6363"` → `163.6363`).

---

## 6. Xử lý dữ liệu không chuẩn

| Tình huống | Hành vi |
| --- | --- |
| Thiếu `paddleocr`/`vietocr`/`torch` | Log warning, tự fallback sang `tesseract` |
| `paddleocr` 2.x vs 3.x khác API | Thử lần lượt `TextDetection.predict` → `PaddleOCR.predict` → `ocr(det, rec, cls)`; thất bại thì báo lỗi kèm **toàn bộ** lỗi từng cách đã thử |
| Backend oneDNN của paddle chết lúc inference | Dựng lại detector với `enable_mkldnn=False` rồi thử lại 1 lần; vẫn lỗi thì fallback sang `tesseract` |
| Không có engine nào | Thoát với exit code 1 kèm hướng dẫn cài đặt (không traceback) |
| OCR ném exception | Bắt lỗi, trả JSON toàn `null` + warning trong `result.warnings` |
| Morphology không dò đủ đường kẻ / lỗi | Fallback PP-Structure → fallback cluster bounding box → fallback raw text |
| PP-Structure lỗi/không tự tin | Fallback cluster bounding box → fallback raw text |
| 2 field cùng khớp một ô bảng (alias trùng) | Giữ field score cao hơn, field còn lại trả `null` kèm warning (trừ field có `keyword` scoping) |
| OCR gộp cả hàng bảng thành một dòng text | Giá trị bị cắt tại nhãn của field kế tiếp; phần dư rỗng vì bị cắt thì **không** lấy dòng dưới (tránh vớ số của cột khác) |
| OCR nhầm ký tự trong số (`1OO`, `l50`) | `fix_ocr_digits` sửa theo ngữ cảnh số (không đụng vào chữ như `8gb`, `150 sms`) |
| Ô/field thiếu | Field đó là `null`, các field khác vẫn được trích xuất |
| Parser ném lỗi | `parse_value` nuốt exception, trả `null` + log warning |
| `schema.yaml` có field lỗi | Bỏ qua field đó kèm warning; schema rỗng hoàn toàn mới raise `SchemaError` |

---

## 7. Giới hạn đã biết (known limitations)

* **VietOCR** cho độ chính xác dấu tiếng Việt tốt hơn recognizer mặc định của
  PaddleOCR, nhưng **vẫn có thể sai** với ảnh chất lượng quá thấp, chữ quá nhỏ,
  hoặc watermark dày đặc đè lên chữ. Tiền xử lý (deskew + adaptive threshold) giảm
  bớt chứ không loại bỏ hoàn toàn vấn đề này.
* **Morphology** (tầng 2, ưu tiên nhất) yêu cầu bảng có **đường kẻ rõ**; bảng
  không viền hoặc đường kẻ quá mờ/đứt sẽ tự fallback PP-Structure. Cột được suy
  từ toạ độ token của **hàng header** — đã kiểm chứng thực nghiệm trên ảnh scan
  thật (200 DPI) là chính xác 8/8 cột cho bảng CTKM mẫu; nếu hàng header bị OCR
  bỏ sót hoàn toàn (không token nào), cột sẽ suy từ toàn bộ token thay thế, độ
  chính xác có thể giảm với bảng có cột hẹp. Token dữ liệu nằm ngoài mọi band
  header được gán theo cột gần nhất — đúng với bảng mà các cột tách bạch rõ,
  nhưng nếu **thứ tự cột trong ảnh không khớp thứ tự band** (bảng có ô gộp ngang
  ở giữa) thì việc gán "gần nhất" này vẫn có thể sai; khi đó nên chạy với
  `--no-morphology` để dùng PP-Structure.
* **Tầng dựng bảng chạy trên ảnh ĐÃ TIỀN XỬ LÝ**, không phải ảnh gốc trên đĩa:
  bounding box của token nằm trong hệ toạ độ ảnh đã deskew/phóng to, nên
  ``build_table`` dùng ``OCRResult.processed_image`` do provider trả về. Trước
  khi sửa, ảnh scan bị nghiêng làm morphology dò được 0 đường kẻ ngang và bị hạ
  cấp xuống cluster bounding box - đúng trên loại ảnh cần nó nhất.
* **Bảng lồng bảng**: nếu ảnh có bảng con nằm trong ô của bảng lớn (biểu mẫu
  BM.12 là ví dụ), tầng morphology chọn dải hàng có **nhiều đường kẻ dọc cắt qua
  nhất** rồi chỉ dựng bảng trong dải đó, và lấy biên cột từ **đường kẻ dọc thật**
  thay vì suy từ token header. Kernel dò đường kẻ dài `0.05` cạnh ảnh (không phải
  `0.35` như bản đầu) để đường kẻ của bảng con không bị phép opening xoá mất.
  Heuristic "dày nhất" giả định bảng cần trích xuất là bảng nhiều cột nhất trang;
  tài liệu có nhiều bảng con ngang nhau sẽ cần `--no-morphology` hoặc PP-Structure.
* **PP-Structure** có thể nhận diện sai với bảng layout bất thường (nhiều bảng lồng
  nhau, đường kẻ đứt, bảng không viền). Đã có fallback cluster bounding box thủ
  công, nhưng cluster thủ công lại dễ sai với **ô merge dọc** — các cơ chế bù trừ
  cho nhau chứ không cơ chế nào đúng tuyệt đối.
* Provider **`tesseract` (fallback) có độ chính xác thấp hơn rõ rệt** provider mặc
  định, đặc biệt với dấu tiếng Việt và ảnh nhiễu. Đây là **đánh đổi có chủ đích**
  giữa *reproducibility* (chỉ cần `apt-get install`, không phải tải hàng trăm MB
  model, chạy được ngay trong môi trường chấm bài không có internet) và *độ chính
  xác*. Khi cần số liệu đúng nhất, hãy dùng `--engine paddle_vietocr`.
* Lần chạy đầu tiên của provider mặc định cần internet để tải model; môi trường
  hoàn toàn offline phải chuẩn bị sẵn cache model hoặc dùng `--engine tesseract`.
* Regex fallback chỉ hoạt động khi nhãn và giá trị nằm **trên cùng một dòng** của
  raw text; bảng có nhãn và giá trị cách nhau quá xa theo chiều dọc cần dựng được
  bảng (tầng 2) mới trích xuất chính xác.
* Chương trình xử lý **một bảng CTKM trên một ảnh**. Ảnh chứa nhiều CTKM sẽ chỉ trả
  về bảng có độ tin cậy cao nhất.
