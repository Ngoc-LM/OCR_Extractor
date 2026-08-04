"""Hai bản Python và C++ phải dùng CÙNG một schema.

Toàn bộ tri thức về mẫu bảng nằm trong schema, nên schema lệch nhau nghĩa là hai
bản cho kết quả khác nhau trên cùng một ảnh - đúng điều README cam kết là không
xảy ra. Đã xảy ra thật một lần: ``schema.json`` có thêm alias
``"Phí đăng ký (VNĐ/tháng)"`` cho ``registerFee``, khiến phần đuôi
``"(VNĐ/tháng) 0"`` của ô ``Cước TB`` bị coi là nhãn của field khác và bản C++
trả ``monthlyFee = 20`` (lấy từ ``"MP 20p đầu tiên"``) trong khi bản Python trả
``0``.

Test tự bỏ qua khi chỉ cài riêng package Python (không có thư mục ``cpp/``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

#: Các khoá quyết định hành vi trích xuất; phần còn lại (comment, thứ tự khoá)
#: không cần giống nhau.
COMPARED_KEYS = ("name", "type", "parser", "parser_args", "aliases", "regex")

PYTHON_SCHEMA = Path(__file__).resolve().parents[1] / "extraction" / "schema.yaml"
CPP_SCHEMA = Path(__file__).resolve().parents[3] / "cpp" / "schema.json"


def load_cpp_schema(path: Path) -> dict[str, Any]:
    """``schema.json`` viết theo kiểu jsonc (có ``//``) nên phải bỏ comment."""
    text = re.sub(r"^\s*//.*$", "", path.read_text(encoding="utf-8"), flags=re.M)
    return json.loads(text)


def comparable(schema: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: field.get(key) for key in COMPARED_KEYS} for field in schema["fields"]]


@pytest.mark.skipif(not CPP_SCHEMA.exists(), reason="không có bản C++ trong cây thư mục")
class TestSchemaParity:
    def test_hai_schema_khai_bao_field_giong_het_nhau(self) -> None:
        python_fields = comparable(yaml.safe_load(PYTHON_SCHEMA.read_text(encoding="utf-8")))
        cpp_fields = comparable(load_cpp_schema(CPP_SCHEMA))

        assert [field["name"] for field in python_fields] == [
            field["name"] for field in cpp_fields
        ], "thứ tự field khác nhau -> thứ tự khoá trong JSON output khác nhau"

        for python_field, cpp_field in zip(python_fields, cpp_fields):
            assert python_field == cpp_field, (
                f"field {python_field['name']!r} khai báo khác nhau giữa "
                f"schema.yaml và cpp/schema.json"
            )

    def test_settings_giong_nhau(self) -> None:
        python_schema = yaml.safe_load(PYTHON_SCHEMA.read_text(encoding="utf-8"))
        cpp_schema = load_cpp_schema(CPP_SCHEMA)

        assert python_schema["settings"] == cpp_schema["settings"]
