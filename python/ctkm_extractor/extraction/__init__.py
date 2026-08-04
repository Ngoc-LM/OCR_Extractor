"""Tầng trích xuất field: schema-driven, parser registry, orchestrator."""

from __future__ import annotations

from .extractor import (
    DEFAULT_SCHEMA_PATH,
    CTKMExtractor,
    ExtractionResult,
    FieldResult,
    FieldSpec,
    Schema,
    SchemaError,
    SchemaSettings,
    extract_image_to_json,
    load_schema,
    match_alias,
)
from .field_parsers import (
    PARSERS,
    available_parsers,
    cycle_parser,
    fold_accents,
    gb_parser,
    get_parser,
    int_parser,
    list_parser,
    money_parser,
    normalize_label,
    parse_value,
    register,
    text_parser,
)

__all__ = [
    "CTKMExtractor",
    "DEFAULT_SCHEMA_PATH",
    "ExtractionResult",
    "FieldResult",
    "FieldSpec",
    "PARSERS",
    "Schema",
    "SchemaError",
    "SchemaSettings",
    "available_parsers",
    "cycle_parser",
    "extract_image_to_json",
    "fold_accents",
    "gb_parser",
    "get_parser",
    "int_parser",
    "list_parser",
    "load_schema",
    "match_alias",
    "money_parser",
    "normalize_label",
    "parse_value",
    "register",
    "text_parser",
]
