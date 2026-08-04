# OCR Extractor — trích xuất chương trình khuyến mại (CTKM) từ ảnh ra JSON

Đọc **1 ảnh** chứa bảng mô tả chương trình khuyến mại viễn thông và trích xuất ra
JSON 11 field:

```bash
cd python
pip install -r requirements.txt
python -m ctkm_extractor.cli --image ../sample.png --out result.json
```

`sample.png` nằm sẵn ở gốc repo (sinh lại được bằng `python tools/make_sample.py`),
nên lệnh trên chạy được ngay sau khi clone và phải ra đúng khối JSON dưới đây.

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

---

## Đối chiếu với đề bài

**Phần lõi theo đúng đề bài nằm ở [`python/`](python/)** — đọc 1 ảnh, xuất JSON 11
field. Mọi thứ khác (bản C++, xử lý PDF nhiều trang, notebook Colab, CI) là **mở
rộng thêm**, không thay thế và không che phần lõi.

| Ràng buộc của đề bài | Cách đáp ứng |
| --- | --- |
| **Không hard-code dữ liệu theo ảnh mẫu** | Toàn bộ tri thức về mẫu bảng nằm trong [`schema.yaml`](python/ctkm_extractor/extraction/schema.yaml); code không chứa giá trị nào của ảnh mẫu. Kiểm được bằng máy: bóc hết docstring rồi tìm `CTKMN180X`/`163636`/`Basic+` trong mọi chuỗi ký tự của code → **0 kết quả** |
| **Dễ mở rộng cho mẫu CTKM khác** | Thêm field mới = thêm 4 dòng YAML, **0 dòng code**. Từ đồng nghĩa, parser, regex fallback đều khai báo trong schema; registry có 9 parser dùng lại được |
| **Xử lý OCR sai hoặc dữ liệu thiếu** | Không bao giờ crash: input rỗng / file không tồn tại / file hỏng đều trả **đủ 11 field `null` + cảnh báo**. Tầng bảng có 4 mức fallback, mỗi field thử 3 chiến lược trước khi bỏ cuộc |
| **Source code + hướng dẫn build/run** | README riêng cho từng bản, 2 notebook Colab chạy được ngay, CI build trên 2 phiên bản Ubuntu |

| Tiêu chí đánh giá | Số đo |
| --- | --- |
| **Độ chính xác** | `sample.png` (ảnh sạch, kèm repo): **11/11**, hai bản ra JSON giống nhau từng byte. Hồ sơ BM.12 thật 6 trang (scan có watermark, bảng lồng bảng, bảng CTKM chỉ nằm ở trang 3): **10/11 giá trị đúng** ở cả hai bản — trượt `spotifyGB` vì OCR đọc "Spotify" thành "Spoúify". Con số đếm **field có giá trị đúng**, không phải field khác null — xem [ghi chú về chọn tiền xử lý](#chọn-tiền-xử-lý-tự-động-và-giới-hạn-của-nó) |
| **Thiết kế & cấu trúc** | 4 tầng tách rời — OCR → dựng bảng → trích xuất theo schema → CLI. Mỗi tầng test được độc lập, tầng dưới không biết gì về tầng trên |
| **Xử lý dữ liệu không chuẩn** | Xem [mục 6 của `python/README.md`](python/README.md) — 13 tình huống đã gặp thật và cách xử lý từng cái |
| **Chất lượng code & mở rộng** | **189 test pytest + 106 test Catch2**, CI chặn mọi cảnh báo trình biên dịch |

### Chọn tiền xử lý tự động và giới hạn của nó

Mặc định chương trình OCR ảnh **hai lần** (có và không nhị phân hoá) rồi giữ lần
trích được nhiều field hơn. Hàm chấm điểm đếm **số field có giá trị**, mà số này
không phân biệt được giá trị đúng với giá trị rác: một lần OCR nhiễu vẫn có thể
điền đủ ô nhưng sai nội dung.

Đo được trên bản scan BM.12 1 trang thu nhỏ còn 50% (ảnh mờ hơn, OCR sai nhiều hơn):

| Cấu hình | Field có giá trị | Field **đúng** |
| --- | --- | --- |
| nhị phân hoá | 7 | 5 |
| không nhị phân hoá | 6 | **6** |

Cơ chế tự chọn lấy nhánh 7 field và mất một field đúng. Nhãn cột vẫn khớp điểm
cao vì rác nằm trong **ô giá trị** chứ không phải ở nhãn, nên điểm khớp alias
không cứu được. Khi đã biết trước loại ảnh thì ép bằng `--no-binarize` /
`--binarize` cho chắc.

Đã thử thay tiêu chí chọn bằng hai tín hiệu khác, đo trên **22 biến thể suy giảm**
(mờ, xoay, thu nhỏ, nhiễu, nhạt/đậm, JPEG hỏng) của hai bản scan — cột cuối là số
field **đúng** bị mất so với lựa chọn tốt nhất có thể:

| Tiêu chí chọn cấu hình | Field đúng bị mất |
| --- | --- |
| đếm field có giá trị (đang dùng) | **1** |
| confidence trung bình của OCR | 6 |
| lọc giá trị có chuỗi thô nhiều ký tự rác | 1 (không cải thiện) |

Confidence không dùng được vì nó lệch **hệ thống**: ảnh nhị phân hoá bị chấm thấp
hơn ~0.11 (trung vị) bất kể chất lượng nhận dạng thật — chỉ đảo dấu ở 4/22 ảnh,
đều là ảnh quá nhạt hoặc quá đậm. So confidence giữa hai cấu hình là so hai thang
điểm khác nhau. Tỉ lệ ký tự rác thì không tách được giá trị sai khỏi
giá trị đúng (trung vị của cả hai nhóm đều bằng 0). Vì vậy giữ nguyên cách đếm
field, và ghi rõ giới hạn ở đây thay vì đổi sang một tiêu chí tệ hơn.

Một điểm **cố tình lệch khỏi ví dụ trong đề**: `onnetMinutes` trả về **chuỗi**
(`"MP 20p đầu tiên"`) chứ không phải số. Đề bài liệt kê field này nhưng không đưa
vào ví dụ output, còn dữ liệu thật trong biểu mẫu là **mô tả chính sách** chứ
không phải số phút — ép về số sẽ mất nghĩa. Muốn đổi thì sửa `type` và `parser`
của field đó trong `schema.yaml`, không đụng tới code.

---

Repo có **hai bản hiện thực song song và độc lập** — chọn một bản, cài đặt và chạy
được ngay, không cần bản còn lại:

| | [`python/`](python/) | [`cpp/`](cpp/) |
| --- | --- | --- |
| Ngôn ngữ | Python 3.10+ | C++17 |
| OCR mặc định | PaddleOCR PP-OCRv5 detect + VietOCR recognize | Cùng 2 model, export ONNX, chạy qua ONNXRuntime C++ API |
| OCR fallback | Tesseract (`pytesseract`) | Tesseract (`TessBaseAPI`) |
| Dựng bảng | morphology → PP-Structure → cluster bbox → raw text | morphology → *(PP-Structure: stretch goal)* → cluster bbox → raw text |
| Schema | `ctkm_extractor/extraction/schema.yaml` | `schema.json` |
| Nhiều trang | `--pdf` (tự tách trang bằng `pymupdf`) | `--image` lặp lại (render PDF trước bằng `pdftoppm`) |
| Tiền xử lý | Mặc định **tự chọn**: chạy cả hai cấu hình nhị phân hoá rồi giữ kết quả nhiều field hơn | giống hệt |
| Test | 189 test `pytest` | 106 test Catch2 |
| Cài đặt | `pip install -r requirements.txt` | `cmake -B build && cmake --build build -j` |

Hai bản dùng **cùng thuật toán, cùng thứ tự xử lý, cùng cách xử lý edge-case**, và
**cùng một schema** — `schema.yaml` với `schema.json` được một test giữ cho luôn
khai báo y hệt nhau (`tests/test_schema_parity.py`), vì schema lệch nhau là đủ để
hai bản cho kết quả khác nhau trên cùng một ảnh. Chạy trên cùng ảnh bảng CTKM,
**output JSON của hai bản giống nhau từng byte**.

---

## Chạy thử trên Google Colab

Không muốn cài gì trên máy — hai notebook chạy trọn vẹn trên Colab:

| Notebook | Nội dung |
| --- | --- |
| [`notebooks/colab_python.ipynb`](notebooks/colab_python.ipynb) | Bản **Python**: chạy bằng Tesseract → bật `paddle_vietocr` → xử lý PDF nhiều trang |
| [`notebooks/colab_cpp.ipynb`](notebooks/colab_cpp.ipynb) | Bản **C++**: build → chạy bằng Tesseract → export `vietocr.onnx` + bảng ký tự để bật engine mặc định → nhiều trang qua `pdftoppm` |

Colab là máy ảo Ubuntu có `sudo` nên `apt`/`cmake`/`g++` chạy bình thường; build bản
C++ đầy đủ mất ~70 giây trên 2 core.

## Chạy nhanh

### Bản Python

```bash
cd python
pip install -r requirements.txt
python -m ctkm_extractor.cli --image ../sample.png --out result.json
python -m pytest ctkm_extractor/tests -q
```

Chi tiết: [python/README.md](python/README.md)

### Bản C++

```bash
cd cpp
sudo apt install -y cmake g++ libopencv-dev libtesseract-dev libleptonica-dev \
                    tesseract-ocr tesseract-ocr-vie
cmake -B build -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime
cmake --build build -j
./build/ctkm_extractor --image ../sample.png --out result.json
ctest --test-dir build
```

Chi tiết: [cpp/README.md](cpp/README.md)

---

## Thiết kế chung của cả hai bản

* **Không hard-code dữ liệu theo ảnh mẫu.** Toàn bộ tri thức về mẫu bảng (tên
  header, từ đồng nghĩa, parser, regex fallback) nằm trong file schema; thêm field
  mới hoặc hỗ trợ mẫu CTKM khác chỉ cần sửa schema, không sửa code.
* **4 tầng tách biệt, test được độc lập**: OCR → dựng bảng → trích xuất field theo
  schema → CLI.
* **Nhiều mức fallback ở mỗi tầng.** Thiếu model nặng thì tự chuyển sang OCR
  fallback; không dò được đường kẻ bảng thì hạ cấp dần xuống cluster bounding box
  rồi raw text + regex.
* **Không bao giờ crash vì dữ liệu xấu.** OCR sai, thiếu ô, thiếu field, parser
  lỗi — đều trả `null` cho field đó kèm cảnh báo, các field còn lại vẫn được trích
  xuất; JSON output luôn đủ 11 field.

Nếu sửa logic ở một bản, hãy đối chiếu và sửa tương ứng ở bản kia — hai bản được
chủ ý giữ hành vi giống nhau.
