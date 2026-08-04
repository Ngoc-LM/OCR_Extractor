"""Test tầng tách trang PDF. Phần chọn trang không cần pymupdf nên luôn chạy."""

from __future__ import annotations

import pytest

from ctkm_extractor.pdf import looks_like_pdf, parse_page_selection


def test_nhan_dien_duoi_pdf():
    assert looks_like_pdf("ho_so.PDF")
    assert looks_like_pdf("/a/b/c.pdf")
    assert not looks_like_pdf("anh.png")


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("2", [1]),
        ("1,3", [0, 2]),
        ("2-4", [1, 2, 3]),
        ("4-2", [1, 2, 3]),          # viết ngược vẫn hiểu
        ("1, 3 , 5", [0, 2, 4]),
        ("2,2,2", [1]),              # trùng thì chỉ lấy một lần
    ],
)
def test_chon_trang(spec, expected):
    assert parse_page_selection(spec, 5) == expected


def test_bo_trong_thi_lay_moi_trang():
    assert parse_page_selection(None, 3) == [0, 1, 2]
    assert parse_page_selection("   ", 3) == [0, 1, 2]


def test_bo_qua_gia_tri_hong_va_ngoai_pham_vi():
    # Gõ nhầm không được làm hỏng cả lượt chạy.
    assert parse_page_selection("abc,2,99", 3) == [1]
    assert parse_page_selection("99", 3) == [0, 1, 2]   # không còn gì hợp lệ -> mọi trang
