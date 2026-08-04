#!/usr/bin/env python3
"""Sinh ảnh mẫu ``sample.png`` dùng cho lệnh khởi động nhanh trong README.

Ảnh được VẼ RA chứ không kèm sẵn ảnh scan thật, vì hai lý do:

* README hứa một lệnh chạy được ngay sau khi clone; không có ảnh nào trong repo
  thì lệnh đó thất bại.
* Ảnh biểu mẫu thật của khách hàng không nên nằm trong repo.

Nội dung bảng khớp đúng khối JSON ví dụ trong README, nên chạy lệnh khởi động
nhanh phải ra đúng khối JSON đó — README tự kiểm chứng được.

Khổ giấy A4 ở 300 DPI, đúng loại đầu vào mà pipeline được tinh chỉnh cho
(``--dpi`` mặc định của nhánh PDF cũng là 300).

    python tools/make_sample.py [đường_dẫn_ra]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

#: Font ứng viên, thử lần lượt cho tới khi mở được (khác nhau theo distro).
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

#: Bảng NHÃN - GIÁ TRỊ, mỗi field một hàng.
#:
#: Biểu mẫu BM.12 thật xếp 8 field thành 8 cột ngang. Vẽ lại đúng bố cục đó thì
#: mỗi ô chỉ còn ~250px cho cả nhãn lẫn giá trị, cỡ chữ phải hạ xuống ~22px và
#: Tesseract bắt đầu nuốt mất ô một chữ số ("0", "150") lẫn bịa chữ ra từ vùng
#: chỉ có đường kẻ dọc. Ảnh mẫu này vì vậy dùng bố cục dọc: chữ to gấp đôi, mỗi
#: hàng một field. Tầng trích xuất xử lý cả hai bố cục như nhau (dò theo NHÃN,
#: không theo chỉ số cột) - muốn thấy bố cục ngang thật thì xem notebook Colab.
ROWS: tuple[tuple[str, str], ...] = (
    ("Tên mã giá OCS", "CTKMN180X"),
    ("Phí đăng ký (VNĐ/tháng)", "163,636.3636"),
    ("Cước TB (VNĐ/tháng)", "0"),
    ("TK thoại", "MP 20p đầu tiên"),
    ("Thoại ngoại mạng", "150"),
    ("SMS Trong nước", "100"),
    ("Lưu lượng Data đa hướng (TK533)", "60GB/tháng, tối đa 8gb/1 ngày"),
    ("Ưu đãi Youtube", "25GB/tháng"),
    ("Ưu đãi Spotify", "25GB/tháng"),
    ("Chu kỳ gia hạn", "tháng"),
    ("Gói cước chính được đăng kí", "Basic+, Family, Corporate++"),
)

#: Bề ngang A4 ở 300 DPI; chiều cao cắt vừa nội dung.
#:
#: Để dư nửa trang trắng bên dưới thì phép phân tích bố cục của Tesseract lấy
#: chính 12 đường kẻ ngang của bảng làm "dòng chữ" và trả về một chuỗi gạch
#: ngang thay vì nội dung.
PAGE_WIDTH = 2480
PAGE_MARGIN = 160
TABLE_TOP = 120
ROW_HEIGHT = 118
FONT_SIZE = 46
#: Khoảng trắng hai bên chữ trong mỗi ô.
CELL_PADDING = 30
#: Nét kẻ đủ đậm để morphological opening không xoá mất khi dò đường kẻ bảng.
LINE_WIDTH = 4


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Font TrueType đầu tiên mở được."""
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit(
        "Không tìm thấy font TrueType nào để vẽ chữ tiếng Việt.\n"
        "Cài một font Unicode (VD: sudo apt install fonts-dejavu-core) rồi chạy lại."
    )


def build_image() -> Image.Image:
    font = load_font(FONT_SIZE)
    page_height = 2 * TABLE_TOP + len(ROWS) * ROW_HEIGHT
    image = Image.new("RGB", (PAGE_WIDTH, page_height), "white")
    draw = ImageDraw.Draw(image)

    table_width = PAGE_WIDTH - 2 * PAGE_MARGIN
    label_width = max(int(draw.textlength(label, font=font)) for label, _ in ROWS)
    label_width += 2 * CELL_PADDING

    left = PAGE_MARGIN
    split = left + label_width
    right = left + table_width
    bottom = TABLE_TOP + len(ROWS) * ROW_HEIGHT

    # CHỈ vẽ viền ngoài, KHÔNG vẽ đường kẻ ngăn giữa các hàng.
    #
    # Đo được: thêm đủ 12 đường kẻ ngang thì phép phân tích bố cục của Tesseract
    # lấy chính các đường kẻ đó làm "dòng chữ" và trả về một chuỗi gạch ngang
    # ("mm", "1 ———") thay vì nội dung bảng - đúng ở mọi độ dày nét và mọi chiều
    # cao hàng đã thử. Đường kẻ của bản scan thật không gây ra chuyện này vì nó
    # đứt quãng và có nhiễu; nét vẽ hoàn hảo thì lại giống một dòng gạch ngang.
    #
    # Bảng không đủ 3 đường kẻ ngang nên tầng morphology bỏ qua và pipeline dùng
    # mức fallback cluster bounding box - vẫn ra đủ 11 field.
    for y in (TABLE_TOP, bottom):
        draw.line([(left, y), (right, y)], fill="black", width=LINE_WIDTH)
    for x in (left, split, right):
        draw.line([(x, TABLE_TOP), (x, bottom)], fill="black", width=LINE_WIDTH)

    for index, (label, value) in enumerate(ROWS):
        # Căn chữ theo baseline của hàng, chừa lề trái bằng CELL_PADDING.
        y = TABLE_TOP + index * ROW_HEIGHT + (ROW_HEIGHT - FONT_SIZE) // 2 - 8
        draw.text((left + CELL_PADDING, y), label, fill="black", font=font)
        draw.text((split + CELL_PADDING, y), value, fill="black", font=font)

    return image


def main() -> None:
    default = Path(__file__).resolve().parent.parent / "sample.png"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    build_image().save(target)
    print(f"Đã ghi ảnh mẫu vào {target}")


if __name__ == "__main__":
    main()
