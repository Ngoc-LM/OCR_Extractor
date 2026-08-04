# CTKM Extractor (C++) — OCR trích xuất chương trình khuyến mại → JSON

Bản C++ **độc lập hoàn toàn** với bản Python (`../python/`): cài đặt và chạy được
mà không cần Python. Về hành vi thì đây là port 1:1 của bản Python — cùng thuật
toán, cùng thứ tự xử lý, cùng cách xử lý edge-case. Đọc **1 ảnh**
chứa bảng mô tả CTKM viễn thông và trích xuất ra JSON 11 field:

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

Bản Python là **nguồn tham chiếu hành vi duy nhất**: cùng kiến trúc, cùng thứ tự
xử lý từng bước, cùng cách xử lý edge-case. Trên ảnh bảng CTKM sạch, output JSON
của hai bản **giống nhau từng byte**, và tầng morphology cho cùng thống kê
(*12 hàng × 2 cột, gán 64/68 token*).

---

## 1. Vì sao ONNXRuntime thay vì Paddle Inference / Tesseract làm engine chính

Giữ detector của PaddleOCR (PP-OCRv5, thuật toán DB) nhưng **thay recognizer mặc
định bằng VietOCR** (transformer huấn luyện riêng cho tiếng Việt) — độ chính xác
dấu tiếng Việt cao hơn rõ rệt so với recognizer đa ngôn ngữ mặc định hay
Tesseract. Cả hai model export ONNX và chạy qua ONNXRuntime C++ API: nhẹ hơn
nhiều so với build full Paddle Inference C++ SDK (`deploy/cpp_infer` phải build
và link Paddle Inference + OpenCV từ nguồn), đồng thời dùng **cùng trọng số** với
bản Python nên kết quả hai bản tương đồng.

Tesseract được giữ làm **fallback provider**: thiếu `models/det.onnx` hoặc
`models/vietocr.onnx` thì `PaddleVietOcrProvider` báo không khả dụng và factory
tự chuyển sang `TesseractOCR` kèm log warning — chương trình không crash.

---

## 2. Kiến trúc — port 1:1 từng tầng của bản Python

```
cpp/
  CMakeLists.txt
  schema.json                      # tương đương schema.yaml của bản Python
  models/                          # det.onnx, vietocr.onnx (không commit weights)
  src/
    util/Utf8.{hpp,cpp}            # đếm/cắt chuỗi theo KÝ TỰ (bản Python dùng str Unicode)
    util/Log.{hpp,cpp}             # thay module logging
    util/Json.{hpp,cpp}            # đọc schema + ghi output (không phụ thuộc lib ngoài)
    ocr/OCRProvider.hpp/.cpp       # port ocr/base.py: OCRToken, BoundingBox, OCRResult
    ocr/Preprocess.{hpp,cpp}       # port preprocess.py: grayscale -> deskew -> adaptive threshold
    ocr/OnnxDetector.{hpp,cpp}     # port phần detect (PP-OCRv5 DB) + hậu xử lý DB
    ocr/OnnxVietOCR.{hpp,cpp}      # port phần recognize (VietOCR)
    ocr/PaddleVietOcrProvider.*    # ghép detect + crop/nắn phối cảnh + recognize
    ocr/TesseractOCR.{hpp,cpp}     # port tesseract_ocr.py - fallback provider
    ocr/ProviderFactory.*          # port ocr/__init__.py: chọn engine + fallback
    table/Morphology.{hpp,cpp}     # port table/morphology.py - ƯU TIÊN NHẤT
    table/PPStructure.{hpp,cpp}    # mức 2 - STRETCH GOAL, chưa hiện thực (xem §7)
    table/Reconstruct.{hpp,cpp}    # port table/reconstruct.py: Table, TableCell, cluster
    table/TableBuilder.*           # port table/__init__.py: điều phối 4 mức fallback
    extraction/Schema.{hpp,cpp}    # nạp schema.json
    extraction/FieldParsers.*      # registry parser
    extraction/Extractor.*         # port extractor.py: orchestrator
    main.cpp                       # port cli.py
  tests/
    test_field_parsers.cpp
    test_extractor.cpp
    test_morphology.cpp
```

`util/*` và hai file `ProviderFactory` / `TableBuilder` là phần bản Python lấy sẵn
từ thư viện chuẩn (`json`, `yaml`, `logging`, str Unicode) hoặc để trong
`__init__.py`; nội dung logic vẫn là port 1:1.

### Tầng 2 — Table: 4 mức fallback

1. **Morphology** (ưu tiên nhất) — `table/Morphology.cpp`
   * Dò **HÀNG** bằng đường kẻ ảnh: morphological opening với kernel chữ nhật dài
     (`h_len = max(15, width * 0.05)`), lấy vị trí đường kẻ từ hình chiếu mật độ
     pixel (ngưỡng `peak * 0.3`), gộp các vị trí cách nhau ≤ 15px thành 1 đường.
   * Với ảnh có **bảng lồng bảng**, thu hẹp về dải hàng có nhiều đường kẻ dọc cắt
     qua nhất trước khi dựng lưới (§7).
   * Dò **CỘT** từ **đường kẻ dọc thật** của dải hàng đang xét; chỉ khi bảng
     không có đủ kẻ dọc mới suy từ toạ độ token của hàng header.
   * Cần tối thiểu 3 đường kẻ ngang, nếu không thì ném lỗi để rơi xuống mức sau.
2. **PP-Structure** — chưa hiện thực (stretch goal), luôn ném lỗi để rơi xuống mức 3.
3. **Cluster bounding box** thủ công: gom theo y rồi theo x.
4. **Raw text blob**: không dựng bảng, để tầng 3 dò bằng regex.

### Tầng 3 — Extraction: schema-driven

Thứ tự thử cho mỗi field: bảng → từng dòng raw text → regex fallback → `null`.
Sau khi resolve xong toàn bộ field có thêm bước **collision guard** (§7).

---

## 3. Cài đặt

### 3.1. Thư viện hệ thống (Ubuntu)

```bash
sudo apt update
sudo apt install -y cmake g++ libopencv-dev \
                    libtesseract-dev libleptonica-dev tesseract-ocr tesseract-ocr-vie
```

Kiểm tra gói tiếng Việt: `tesseract --list-langs | grep vie`.

Hỗ trợ **cả Tesseract 4 lẫn 5** (Ubuntu 22.04 và Google Colab vẫn dùng bản 4;
Ubuntu 24.04 đã sang bản 5). Code chỉ dùng những API giống hệt nhau ở cả hai đời.
Đáng chú ý là **không** dùng `GetAvailableLanguagesAsVector()` để dò gói ngôn ngữ:
hàm đó đổi chữ ký giữa hai bản (`GenericVector<STRING>*` so với
`std::vector<std::string>*`), mà `GenericVector` lại chỉ được khai báo trước trong
`baseapi.h` nên muốn dùng phải kéo theo header nội bộ đã bị gỡ ở bản 5. Thay vào
đó `pickLanguage()` thử `Init()` với từng gói ứng viên — API ổn định ở mọi đời, và
trả lời đúng câu hỏi cần biết: gói này có **dùng được** không, chứ không chỉ có mặt
trên đĩa. CI build trên cả hai phiên bản Ubuntu để chặn loại lỗi này.

### 3.2. ONNXRuntime prebuilt

```bash
curl -L -o onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xzf onnxruntime.tgz
sudo mv onnxruntime-linux-x64-1.20.1 /opt/onnxruntime
```

CMake tự tìm ở `/opt/onnxruntime`; nơi khác thì truyền
`-DONNXRUNTIME_ROOT_DIR=<đường dẫn>`. **Không có ONNXRuntime vẫn build được** —
provider mặc định báo không khả dụng và tự fallback sang Tesseract.

### 3.3. Chuẩn bị model (`models/`)

Weights **không commit vào repo**. Cần 3 file:

| File | Cách lấy |
| --- | --- |
| `models/det.onnx` | Tải model detect PP-OCRv5 (`PP-OCRv5_mobile_det` hoặc `_server_det`) từ PaddleOCR, rồi `paddle2onnx --model_dir <thư mục inference> --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file models/det.onnx --opset_version 11` |
| `models/vietocr.onnx` | Chạy notebook [`colab_cpp.ipynb`](../notebooks/colab_cpp.ipynb) — có sẵn script export hoàn chỉnh |
| `models/vietocr_vocab.txt` | Cùng notebook trên xuất ra. Mỗi dòng một ký tự, thứ tự đúng bằng id của `vietocr.model.vocab.Vocab`: dòng 0-3 là `<pad> <sos> <eos> *`, từ dòng 4 là `config["vocab"]` |

**Model contract** mà `OnnxVietOCR` giả định (xem đầu `src/ocr/OnnxVietOCR.hpp`):

* Input ảnh `float32 NCHW [1, 3, 32, W]`, giá trị **chia 255** → `[0, 1]` — đúng
  như `vietocr.tool.translate.process_image`. Bề rộng lấy theo tỉ lệ rồi **làm
  tròn lên bội số của 10** (`ceil(w/10)*10`), kẹp trong `[32, 512]`.
* Model **1 input** → output `[1, T, V]` hoặc `[T, 1, V]`: giải mã greedy một lượt.
* Model **2 input** → input thứ hai là chuỗi id đã sinh (`int64 [1, L]`): giải mã
  tự hồi quy, mỗi bước lấy argmax của bước cuối, dừng ở `<eos>`.

Nếu model của bạn export theo layout khác, chỉ cần sửa `prepareCrop()` và vòng
giải mã trong `OnnxVietOCR.cpp`; các tầng khác không đổi.

### 3.4. Build & test

```bash
cmake -B build -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime
cmake --build build -j
ctest --test-dir build
```

Test dùng **Catch2 v3 qua FetchContent** (cần internet lần đầu). Không có mạng thì
`cmake -B build -DCTKM_BUILD_TESTS=OFF`.

### 3.5. Cài đặt (tuỳ chọn)

```bash
cmake --install build --prefix /usr/local
```

Cài `bin/ctkm_extractor` và `share/ctkm_extractor/schema.json`. Binary giữ rpath
tới thư viện đã link (kể cả `libonnxruntime.so` ngoài thư mục hệ thống) nên chạy
được ngay từ bất kỳ đâu; trỏ schema bằng biến môi trường:

```bash
export CTKM_SCHEMA=/usr/local/share/ctkm_extractor/schema.json
ctkm_extractor --image ../sample.png --out result.json
```

---

## 4. Chạy chương trình

```bash
# Mặc định: ONNX detect + VietOCR recognize, dựng bảng bằng morphology
./build/ctkm_extractor --image ../sample.png --out result.json

# In JSON ra stdout
./build/ctkm_extractor --image ../sample.png

# Ép provider cụ thể
./build/ctkm_extractor --image ../sample.png --out result.json --engine paddle_vietocr
./build/ctkm_extractor --image ../sample.png --out result.json --engine tesseract

# Debug: in raw OCR text, bảng đã dựng, nguồn + score từng field ra stderr
./build/ctkm_extractor --image ../sample.png --out result.json --debug

# Xem engine nào khả dụng
./build/ctkm_extractor --list-engines
```

| Cờ | Ý nghĩa |
| --- | --- |
| `--schema <path>` | `schema.json` tuỳ biến thay cho schema đi kèm |
| `--no-morphology` | Bỏ qua dò bảng bằng đường kẻ, dùng thẳng mức fallback sau |
| `--no-pp-structure` | Bỏ qua PP-Structure |
| `--strict-engine` | Không tự fallback sang engine khác khi engine yêu cầu thiếu dependency |
| `--no-binarize` / `--binarize` | Ép TẮT/BẬT adaptive threshold. **Mặc định chạy cả hai rồi giữ kết quả trích được nhiều field hơn**; ép để chỉ chạy một lượt (nhanh gấp đôi) |
| `--image <path>` | **Lặp lại được**: mỗi lần một ảnh/một trang. Nhiều ảnh thì OCR từng ảnh rồi gộp (xem §4.1) |
| `--text-file <path>` | Bỏ qua OCR, đọc raw text từ file — debug riêng tầng trích xuất |
| `--indent <n>` | Số space thụt lề JSON (mặc định 2) |

CLI luôn ghi JSON **đủ 11 field** (thiếu thì `null`) kể cả khi OCR lỗi; exit code
khác 0 chỉ khi lỗi cấu hình (schema hỏng, không tìm thấy ảnh, không có engine).

### 4.1. PDF nhiều trang

Bản C++ **không đọc PDF** (sẽ phải kéo thêm poppler/mupdf). Render trước rồi
truyền từng trang:

```bash
pdftoppm -r 300 -png ho_so.pdf trang        # -> trang-1.png, trang-2.png, ...
./build/ctkm_extractor --image trang-1.png --image trang-2.png --image trang-3.png \
                       --out result.json --debug
```

Bảng CTKM thường **chỉ nằm ở một trang** trong cả tập hồ sơ, nên thuật toán
chọn/gộp trang (`mergePageResults`) **giống hệt bản Python**: chấm điểm mỗi trang
theo *(số field trích được, điểm khớp trung bình)*, lấy trang điểm cao nhất làm
trang chính, chỉ **bù** field còn thiếu từ trang khác chứ không ghi đè. Đo trên
hồ sơ BM.12 thật 6 trang, hai bản cùng chọn trang 3 và cho cùng kết quả.

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

---

## 5. Thêm field / alias mới — chỉ sửa `schema.json`, **không cần build lại**

Chương trình đọc `schema.json` lúc chạy (thứ tự tìm: `--schema` → biến môi trường
`CTKM_SCHEMA` → `./schema.json` → đường dẫn cấu hình lúc build).

```jsonc
{
  "name": "tiktokGB",          // tên khoá trong JSON output
  "type": "number",            // number | string | array
  "parser": "gb_parser",       // tên hàm trong src/extraction/FieldParsers.cpp
  "parser_args": {
    "keyword": "Tiktok",
    "fallback_to_number": true
  },
  "aliases": ["Tiktok", "Data Tiktok", "Ưu đãi Tiktok"],
  "regex": ["(tiktok[^0-9\\n]{0,30}[0-9][0-9.,]*\\s*g\\s?b)"]
}
```

Mẫu CTKM khác thường chỉ khác **tên cột / thứ tự cột**: thêm cách viết đó vào
`aliases` của field tương ứng. Thứ tự cột không quan trọng — extractor dò theo
nhãn chứ không theo chỉ số cột.

**Ghi chú về `regex`**: pattern được so khớp trên bản text **đã bỏ dấu**, nên viết
pattern không dấu và **chỉ dùng ASCII**; giá trị trả về vẫn cắt từ text gốc còn
dấu (nhờ bảng ánh xạ vị trí trong `FoldedText`). Nhóm bắt (capture group) số 1
phải là **giá trị** — dùng `(?:...)` cho các nhóm lựa chọn của phần nhãn.

### Parser có sẵn

| Parser | Chức năng | Ví dụ |
| --- | --- | --- |
| `money_parser` | Số tiền lẫn lộn `,` và `.` | `"163,636.3636"` → `163636.3636`; `"150.534.213"` → `150534213` |
| `int_parser` | Số nguyên đầu tiên | `"MP 20p đầu tiên"` → `20`; `"1.500 phút"` → `1500` |
| `gb_parser` | Dung lượng GB đầu tiên | `"60GB/tháng, tối đa 8gb/1 ngày"` → `60` |
| `cycle_parser` | Từ khoá chu kỳ | `"Chu kỳ gia hạn: tháng"` → `"tháng"` |
| `list_parser` | Tách danh sách theo dấu phẩy | `"Basic+, Family, Corporate++"` → `["Basic+", "Family", "Corporate++"]` |
| `text_parser` | Chuỗi đã làm sạch, lọc theo pattern | `" : ctkmn180x "` → `"CTKMN180X"` |

Heuristic của `money_parser`: có **cả** `,` và `.` → dấu **sau cùng** là thập
phân; chỉ một loại dấu lặp lại và mọi nhóm sau đều đúng 3 chữ số → **tất cả là
hàng nghìn** (`"150.534.213"` → `150534213`, **không phải** `150534.213`); chỉ một
dấu duy nhất → hàng nghìn nếu nhóm sau đúng 3 chữ số, ngược lại là thập phân.

---

## 6. Test

```bash
ctest --test-dir build            # 106 test
./build/ctkm_tests "[regression]" # chỉ chạy nhóm regression
```

* `tests/test_field_parsers.cpp` — mọi edge-case của `money_parser`, `text_parser`
  cho `onnetMinutes`, dữ liệu OCR nhiễu/thiếu không được crash.
* `tests/test_extractor.cpp` — end-to-end với OCR mock; **có test dùng đúng nguyên
  văn 8 header của ảnh mẫu thật (BM.12)** (`"Tên mã giá OCS"`,
  `"Phí đăng ký (VNĐ/tháng)"`, `"Cước TB (VND tháng)"`, `"TK thoại"`,
  `"Thoại ngoại mạng"`, `"SMS Trong nước"`, `"Lưu lượng Data đa hướng (TK533)"`,
  `"Ưu đãi data đơn"`) + test collision guard.
* `tests/test_morphology.cpp` — ảnh bảng viền tổng hợp vẽ bằng OpenCV: dò đúng số
  hàng/cột, token tràn ngoài band vẫn vào đúng cột, hàng cuối thiếu viền dưới vẫn
  được giữ, chữ chân trang không bị kéo vào bảng.

---

## 7. Giới hạn đã biết

* **PP-Structure chưa được port** (stretch goal): mức fallback 2 luôn ném lỗi nên
  morphology thất bại là rơi thẳng xuống cluster bounding box. Bảng CTKM có đường
  kẻ rõ nên mức 1 đã xử lý tốt; bảng **không viền** sẽ mất một lớp dự phòng.
* **Tầng dựng bảng chạy trên ảnh ĐÃ TIỀN XỬ LÝ**, không phải ảnh gốc trên đĩa:
  bounding box của token nằm trong hệ toạ độ ảnh đã deskew/phóng to, nên
  `buildTable` dùng `OCRResult::processedImage` do provider trả về. Trước khi
  sửa, ảnh scan bị nghiêng làm morphology dò được 0 đường kẻ ngang và bị hạ cấp
  xuống cluster bounding box - đúng trên loại ảnh cần nó nhất.
* **Bảng lồng bảng**: nếu ảnh có bảng con nằm trong ô của bảng lớn (biểu mẫu
  BM.12 là ví dụ), tầng morphology chọn dải hàng có **nhiều đường kẻ dọc cắt qua
  nhất** rồi chỉ dựng bảng trong dải đó, và lấy biên cột từ **đường kẻ dọc thật**
  thay vì suy từ token header. Kernel dò đường kẻ dài `0.05` cạnh ảnh (không phải
  `0.35` như bản đầu) để đường kẻ của bảng con không bị phép opening xoá mất.
  Heuristic "dày nhất" giả định bảng cần trích xuất là bảng nhiều cột nhất trang;
  tài liệu có nhiều bảng con ngang nhau sẽ cần `--no-morphology` hoặc PP-Structure.
* **Cột suy từ token hàng header**: nếu tính band cột từ toạ độ x của *tất cả*
  token, giá trị dài ở hàng dữ liệu làm 2 band kề nhau bị nối nhầm thành 1 (thực
  nghiệm trên ảnh scan thật: gộp mọi hàng ra sai 6/8 cột, chỉ dùng hàng header ra
  đúng 8/8 cột). Đổi lại, nếu hàng header bị OCR bỏ sót hoàn toàn thì phải suy cột
  từ toàn bộ token, độ chính xác giảm với bảng có cột hẹp.
* **Token tràn ra ngoài mọi band cột** không chồng lấn band nào; phải chọn band
  **gần nhất theo khoảng cách** thay vì để rơi về cột 0 — nếu không, ô nhãn sẽ nuốt
  giá trị (`"Ưu đãi Data"` nuốt `"tối đa 8gb/1 ngày"` → `dataGB` ra 8 thay vì 60).
* **Hàng cuối chưa đóng viền**: ảnh scan/cắt cúp hay thiếu đường kẻ dưới cùng;
  token dưới đường kẻ cuối trong phạm vi 1 bước hàng vẫn được giữ, xa hơn thì loại
  (tránh kéo chữ chân trang vào bảng).
* **`onnetMinutes` là string, không phải number**: giá trị thật là mô tả chính sách
  (`"MP 20p đầu tiên"`). Ép về number sẽ luôn `null`, hoặc do alias fuzzy-match sẽ
  lấy nhầm giá trị của `offnetMinutes`.
* **Thứ tự candidate khi tra ô bảng**: chỉ khi nhãn nằm ở hàng header của bảng
  **thực sự nhiều cột** (`n_cols > 1`) mới thử ô liền kề trước phần dư trong ô
  nhãn. Nếu không phân biệt bằng `n_cols > 1` sẽ phá vỡ bảng 1 cột dựng từ text thô
  (`"Tên mã giá OCS CTKMN180X"` trên cùng 1 dòng bị đọc nhầm sang dòng kế tiếp).
* **Collision guard**: 2 field cùng lấy giá trị từ một ô bảng thì chỉ field có score
  cao hơn được giữ, field còn lại trả `null` kèm cảnh báo — trừ field khai báo
  `keyword` trong `parser_args` (`youtubeGB`/`spotifyGB` cùng đọc từ 1 ô gộp là
  thiết kế có chủ đích).
* **OCR gộp cả hàng bảng thành một dòng text**: phần dư sau nhãn khi đó chứa luôn
  mọi cột phía sau, nên `cutAtNextLabel()` cắt giá trị tại vị trí nhãn của field
  **khác** bắt đầu. Không cắt thì trên BM.12 thật, `monthlyFee` (ô giá trị bị OCR
  bỏ sót) vớ phải `533` trong `"(TK533)"` — mã tài khoản của cột Data. Alias ngắn
  hơn 5 ký tự không làm mốc cắt và chỉ cắt ở **ranh giới từ**. Phần dư rỗng *vì bị
  cắt* thì **không** được lấy dòng kế tiếp làm giá trị (chỉ nhãn chiếm trọn dòng
  mới được), nếu không lại vớ phải nội dung của cột khác.
* **Regex chỉ dùng cho ASCII**: `std::regex` làm việc trên byte nên không dùng để
  match cụm từ có dấu tiếng Việt; so khớp header/alias dùng `std::string::find` và
  fuzzy-match trên chuỗi đã bỏ dấu (an toàn trên UTF-8 vì byte tiếp diễn không
  trùng byte ASCII).
* **`min_item_length` / `max_length` tính theo ký tự, `{n,m}` trong regex tính theo
  byte**: các tham số parser dùng `utf8::length` để khớp bản Python, còn số lượng
  trong pattern regex thì đếm byte — với chuỗi tiếng Việt, `[^\n]{1,60}` là 60 byte
  chứ không phải 60 ký tự.
* **`minAreaRect().angle` khác nhau giữa các bản OpenCV**: đo trên cùng một ảnh
  nghiêng 2°, OpenCV 4.6 trả `2.02` còn OpenCV 5.0 trả `-87.98`. Vì vậy
  `estimateSkewAngle` chuẩn hoá góc bằng **cả hai nhánh** (`> 45` thì trừ 90,
  `< -45` thì cộng 90) — bản Python cũng vậy. Thiếu nhánh cộng 90 thì trên OpenCV 5
  mọi ảnh nghiêng nhẹ đều bị coi là lệch gần 90° và bỏ qua deskew, khiến hai bản
  cho text OCR khác nhau trên ảnh nhiễu nặng.
* **Tesseract (fallback) độ chính xác thấp hơn provider mặc định**, đặc biệt với
  dấu tiếng Việt và ảnh nhiễu — đánh đổi có chủ đích giữa reproducibility (chỉ cần
  `apt install`, không tải model nặng) và độ chính xác.
