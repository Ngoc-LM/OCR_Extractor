"""Orchestrator: OCR -> dựng bảng -> map field theo schema -> parse -> JSON.

Luồng xử lý cho mỗi field, dừng ở bước đầu tiên cho kết quả:

1. Tìm nhãn (alias) trong **bảng** đã dựng, lấy ô giá trị bên phải/bên dưới.
2. Tìm nhãn theo **từng dòng** của raw text (khi bảng thiếu ô hoặc không dựng được).
3. Dò **regex fallback** khai báo trong ``schema.yaml`` trên toàn bộ raw text.
4. Không tìm được -> ghi log warning và trả ``None`` (không raise).
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field as dataclass_field, replace as dataclasses_replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ocr import OCRProvider, OCRResult, PreprocessConfig, create_provider
from ..table import (
    STRATEGY_RAW_TEXT,
    Table,
    TableBuildResult,
    build_table,
)
from .field_parsers import fold_accents, normalize_label, parse_value

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schema.yaml")

SOURCE_TABLE = "table"
SOURCE_TEXT_LINE = "text_line"
SOURCE_REGEX = "regex"
SOURCE_MISSING = "missing"

#: Ký tự phân tách nhãn với giá trị khi cả hai nằm trong cùng một ô.
LABEL_VALUE_SEPARATORS = ":-–—|="

#: Alias ngắn hơn chừng này ký tự không được dùng làm mốc cắt giá trị: alias
#: quá ngắn (``"SMS"``, ``"Data"``) dễ khớp nhầm vào giữa một cụm từ bình thường
#: và cắt cụt giá trị hợp lệ.
MIN_BOUNDARY_ALIAS_LENGTH = 5


class SchemaError(RuntimeError):
    """Schema không hợp lệ hoặc không đọc được."""


@dataclass
class FieldSpec:
    """Khai báo của một field trong ``schema.yaml``."""

    name: str
    parser: str = "text_parser"
    type: str = "string"
    aliases: list[str] = dataclass_field(default_factory=list)
    regex: list[str] = dataclass_field(default_factory=list)
    parser_args: dict[str, Any] = dataclass_field(default_factory=dict)
    required: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldSpec":
        """Dựng ``FieldSpec`` từ một mục trong schema, bỏ qua khoá lạ."""
        name = str(payload.get("name") or "").strip()
        if not name:
            raise SchemaError(f"Field thiếu thuộc tính 'name': {payload!r}")
        regex = payload.get("regex") or []
        if isinstance(regex, str):
            regex = [regex]
        aliases = payload.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        return cls(
            name=name,
            parser=str(payload.get("parser") or "text_parser"),
            type=str(payload.get("type") or "string"),
            aliases=[str(a) for a in aliases],
            regex=[str(r) for r in regex],
            parser_args=dict(payload.get("parser_args") or {}),
            required=bool(payload.get("required", False)),
        )


@dataclass
class SchemaSettings:
    """Tham số điều khiển việc so khớp nhãn."""

    header_rows: int = 3
    min_similarity: float = 0.82
    max_value_distance: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SchemaSettings":
        data = payload or {}
        defaults = cls()
        try:
            return cls(
                header_rows=int(data.get("header_rows", defaults.header_rows)),
                min_similarity=float(data.get("min_similarity", defaults.min_similarity)),
                max_value_distance=int(
                    data.get("max_value_distance", defaults.max_value_distance)
                ),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("settings trong schema không hợp lệ (%s), dùng mặc định", exc)
            return defaults


@dataclass
class Schema:
    """Toàn bộ schema đã nạp."""

    fields: list[FieldSpec]
    settings: SchemaSettings = dataclass_field(default_factory=SchemaSettings)
    version: int = 1
    path: str | None = None

    @property
    def field_names(self) -> list[str]:
        return [spec.name for spec in self.fields]


def load_schema(path: str | Path | None = None) -> Schema:
    """Nạp schema từ file YAML.

    :raises SchemaError: khi file không tồn tại, YAML hỏng, hoặc thiếu mục ``fields``.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise SchemaError(f"Thiếu PyYAML để đọc schema: {exc}") from exc

    schema_path = Path(path) if path else DEFAULT_SCHEMA_PATH
    if not schema_path.is_file():
        raise SchemaError(f"Không tìm thấy schema: {schema_path}")
    try:
        payload = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SchemaError(f"Không đọc được schema {schema_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"Schema {schema_path} phải là một mapping YAML")

    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise SchemaError(f"Schema {schema_path} thiếu danh sách 'fields'")

    fields: list[FieldSpec] = []
    for item in raw_fields:
        if not isinstance(item, dict):
            logger.warning("Bỏ qua mục field không hợp lệ trong schema: %r", item)
            continue
        try:
            fields.append(FieldSpec.from_dict(item))
        except SchemaError as exc:
            logger.warning("Bỏ qua field lỗi trong schema: %s", exc)
    if not fields:
        raise SchemaError(f"Schema {schema_path} không có field hợp lệ nào")

    return Schema(
        fields=fields,
        settings=SchemaSettings.from_dict(payload.get("settings")),
        version=int(payload.get("version", 1) or 1),
        path=str(schema_path),
    )


@dataclass
class FieldResult:
    """Kết quả trích xuất của một field."""

    name: str
    value: Any = None
    raw_value: str | None = None
    source: str = SOURCE_MISSING
    score: float = 0.0
    matched_alias: str | None = None
    #: Toạ độ ô bảng đã cung cấp giá trị (chỉ có khi source == SOURCE_TABLE),
    #: dùng để phát hiện khi 2 field vô tình cùng lấy chung một ô.
    cell_coords: tuple[int, int] | None = None
    #: Trang PDF (1-based) đã cho ra giá trị này; None khi đầu vào chỉ có 1 ảnh.
    source_page: int | None = None

    @property
    def found(self) -> bool:
        return self.value is not None


@dataclass
class ExtractionResult:
    """Kết quả trích xuất toàn bộ ảnh/khối text."""

    fields: dict[str, FieldResult] = dataclass_field(default_factory=dict)
    raw_text: str = ""
    table: Table | None = None
    table_strategy: str = STRATEGY_RAW_TEXT
    ocr_provider: str = "unknown"
    warnings: list[str] = dataclass_field(default_factory=list)
    #: Trang PDF (1-based) được chọn làm trang chính; None khi đầu vào là 1 ảnh.
    page: int | None = None
    #: Số trang đã OCR (chỉ khác 0 khi đầu vào là PDF nhiều trang).
    pages_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON output: đúng thứ tự field trong schema, thiếu thì ``None``."""
        return {name: result.value for name, result in self.fields.items()}

    def to_json(self, indent: int = 2) -> str:
        """Chuỗi JSON UTF-8 (không escape tiếng Việt)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def debug_report(self) -> str:
        """Báo cáo chi tiết dùng cho ``--debug``."""
        lines: list[str] = []
        lines.append(f"OCR provider     : {self.ocr_provider}")
        lines.append(f"Table strategy   : {self.table_strategy}")
        if self.pages_processed:
            lines.append(
                f"Trang            : {self.page} (chính) trong {self.pages_processed} trang đã OCR"
            )
        if self.table is not None:
            lines.append(
                f"Table size       : {self.table.n_rows} hàng x {self.table.n_cols} cột "
                f"(confidence={self.table.confidence:.2f})"
            )
        lines.append("")
        lines.append("--- OCR raw text ---")
        lines.append(self.raw_text or "(rỗng)")
        lines.append("")
        lines.append("--- Bảng đã dựng ---")
        lines.append(self.table.render() if self.table is not None else "(không dựng được bảng)")
        lines.append("")
        lines.append("--- Nguồn từng field ---")
        for name, result in self.fields.items():
            lines.append(
                f"{name:<16} = {result.value!r:<28} "
                f"[source={result.source}, score={result.score:.2f}, raw={result.raw_value!r}"
                + (f", trang={result.source_page}]" if result.source_page is not None else "]")
            )
        if self.warnings:
            lines.append("")
            lines.append("--- Cảnh báo ---")
            for warning in self.warnings:
                lines.append(f"* {warning}")
        return "\n".join(lines)


def similarity(left: str, right: str) -> float:
    """Độ tương đồng 0..1 giữa hai nhãn đã chuẩn hoá."""
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def match_alias(text: str, aliases: Sequence[str], min_similarity: float) -> tuple[float, str | None]:
    """Điểm khớp tốt nhất giữa nội dung một ô và danh sách alias của field.

    Trả ``(score, alias)``; ``score < min_similarity`` nghĩa là không khớp.
    """
    normalized_text = normalize_label(text)
    if not normalized_text:
        return 0.0, None
    best_score = 0.0
    best_alias: str | None = None
    for alias in aliases:
        normalized_alias = normalize_label(alias)
        if not normalized_alias:
            continue
        if normalized_text == normalized_alias:
            score = 1.0
        elif normalized_text.startswith(normalized_alias):
            score = 0.97
        elif normalized_alias in normalized_text:
            # Nhãn nằm lẫn trong câu dài thì độ tin cậy giảm theo tỉ lệ độ dài.
            ratio = len(normalized_alias) / len(normalized_text)
            score = 0.85 + 0.1 * ratio
        else:
            score = similarity(normalized_text, normalized_alias)
        if score > best_score:
            best_score = score
            best_alias = alias
    if best_score < min_similarity:
        return best_score, None
    return best_score, best_alias


def remainder_after_alias(text: str, alias: str) -> str:
    """Phần còn lại của ô sau khi bỏ nhãn (``"Chu kỳ: tháng"`` -> ``"tháng"``)."""
    folded_text = fold_accents(text or "").lower()
    folded_alias = fold_accents(alias or "").lower()
    position = folded_text.find(folded_alias)
    if position < 0 or not folded_alias:
        return ""
    tail = text[position + len(folded_alias) :]
    return tail.strip().lstrip(LABEL_VALUE_SEPARATORS).strip()


def _is_word_boundary(folded: str, start: int, length: int) -> bool:
    """True nếu ``folded[start:start+length]`` là một cụm từ trọn vẹn."""
    before = folded[start - 1] if start > 0 else " "
    after_index = start + length
    after = folded[after_index] if after_index < len(folded) else " "
    return not before.isalnum() and not after.isalnum()


def cut_at_next_label(
    value: str, own_aliases: Sequence[str], other_aliases: Sequence[str]
) -> str:
    """Cắt giá trị tại vị trí nhãn của **field khác** bắt đầu.

    Cần thiết vì OCR trả về nguyên một HÀNG của bảng thành MỘT dòng text: phần
    còn lại của dòng sau nhãn chứa luôn mọi cột phía sau nó. Không cắt thì
    parser vớ phải số của cột khác - đo trên biểu mẫu BM.12 thật, ô giá trị của
    ``"Cước TB"`` bị OCR bỏ sót nên field ``monthlyFee`` nhận cả phần đuôi
    ``"Thoại ngoại mạng ... (TK533) ..."`` và trả ra ``533``, là số của một cột
    hoàn toàn khác. Quy tắc "giá trị thuộc về nhãn cho tới khi nhãn kế tiếp bắt
    đầu" là quy ước chung khi bóc tách văn bản có nhãn, không gắn với tài liệu
    cụ thể nào.

    Alias của **chính field đang xét** không cắt (nhãn có thể lặp lại trong
    chính phần giá trị), và alias quá ngắn cũng không dùng làm mốc vì dễ khớp
    nhầm vào giữa một cụm từ bình thường.
    """
    if not value:
        return ""
    folded_value = fold_accents(value).lower()
    own = {fold_accents(alias).lower() for alias in own_aliases}
    cut = len(value)
    for alias in other_aliases:
        folded_alias = fold_accents(alias or "").lower().strip()
        if len(folded_alias) < MIN_BOUNDARY_ALIAS_LENGTH or folded_alias in own:
            continue
        position = folded_value.find(folded_alias)
        while 0 <= position < cut:
            if _is_word_boundary(folded_value, position, len(folded_alias)):
                cut = position
                break
            position = folded_value.find(folded_alias, position + 1)
    if cut >= len(value):
        return value
    return value[:cut].strip().strip(LABEL_VALUE_SEPARATORS).strip()


def page_extraction_score(result: "ExtractionResult") -> tuple[int, float]:
    """Điểm của một trang: (số field trích được, điểm khớp trung bình).

    Trang không có bảng CTKM sẽ ra ``(0, 0.0)`` nên không bao giờ được chọn làm
    trang chính khi tồn tại trang khác có dữ liệu.
    """
    found = [item for item in result.fields.values() if item.found]
    if not found:
        return (0, 0.0)
    return (len(found), sum(item.score for item in found) / len(found))


def merge_page_results(
    pages: Sequence[tuple[int, "ExtractionResult"]],
) -> "ExtractionResult":
    """Gộp kết quả của nhiều trang thành một kết quả duy nhất.

    Bảng CTKM thường chỉ nằm ở MỘT trang trong cả tập hồ sơ, nên chiến lược là:

    1. Chọn **trang chính** = trang trích được nhiều field nhất (hoà thì so điểm
       khớp trung bình, vẫn hoà thì lấy trang có số nhỏ hơn).
    2. Lấy toàn bộ giá trị của trang chính làm nền.
    3. Field nào **vẫn thiếu** mới lấy bù từ các trang còn lại, xét theo thứ tự
       điểm giảm dần, và ghi rõ trong cảnh báo là lấy từ trang nào.

    Bước 3 cố tình **không** ghi đè giá trị đã có: trang chính là trang thật sự
    chứa bảng, còn trang khác dễ có nhãn trùng tên trong phần văn bản thường.
    """
    if not pages:
        return ExtractionResult(pages_processed=0)

    ranked = sorted(
        pages, key=lambda item: (-page_extraction_score(item[1])[0],
                                 -page_extraction_score(item[1])[1], item[0])
    )
    primary_page, primary = ranked[0]

    merged = ExtractionResult(
        fields={
            name: dataclasses_replace(
                item, source_page=primary_page if item.found else None
            )
            for name, item in primary.fields.items()
        },
        raw_text=primary.raw_text,
        table=primary.table,
        table_strategy=primary.table_strategy,
        ocr_provider=primary.ocr_provider,
        warnings=list(primary.warnings),
        page=primary_page,
        pages_processed=len(pages),
    )

    for page_number, result in ranked[1:]:
        for name, item in result.fields.items():
            current = merged.fields.get(name)
            if current is not None and current.found:
                continue
            if not item.found:
                continue
            merged.fields[name] = dataclasses_replace(item, source_page=page_number)
            message = f"Field '{name}' lấy từ trang {page_number} (trang chính là {primary_page})"
            logger.info(message)
            merged.warnings.append(message)

    empty = [str(number) for number, result in pages if page_extraction_score(result)[0] == 0]
    if empty:
        logger.info("Trang không có dữ liệu CTKM: %s", ", ".join(empty))
    return merged


class CTKMExtractor:
    """Điều phối toàn bộ pipeline trích xuất CTKM."""

    def __init__(
        self,
        schema: Schema | str | Path | None = None,
        *,
        provider: OCRProvider | None = None,
        engine: str = "paddle_vietocr",
        use_morphology: bool = True,
        use_pp_structure: bool = True,
        strict_engine: bool = False,
        provider_kwargs: dict[str, Any] | None = None,
        binarize: bool | None = None,
    ) -> None:
        """Khởi tạo extractor.

        :param schema: đối tượng :class:`Schema`, đường dẫn YAML, hoặc ``None`` để
            dùng ``extraction/schema.yaml`` mặc định.
        :param provider: OCR provider dựng sẵn (test hay inject mock thì truyền vào đây).
        :param engine: tên engine dùng khi ``provider`` không được truyền.
        :param use_morphology: đặt False để bỏ qua mức 1 (dò đường kẻ bảng bằng
            CV cổ điển) - hữu ích khi test hoặc khi biết trước ảnh không có
            đường viền bảng rõ.
        :param strict_engine: True thì không fallback sang engine khác khi thiếu
            dependency.
        """
        if isinstance(schema, Schema):
            self.schema = schema
        else:
            self.schema = load_schema(schema)
        self._provider = provider
        self.engine = engine
        self.use_morphology = use_morphology
        self.use_pp_structure = use_pp_structure
        self.strict_engine = strict_engine
        self.provider_kwargs = dict(provider_kwargs or {})
        #: ``True``/``False`` = ép bật/tắt nhị phân hoá; ``None`` = TỰ CHỌN
        #: (chạy cả hai rồi giữ kết quả trích được nhiều field hơn).
        self.binarize = binarize

    # ------------------------------------------------------------------
    # Điểm vào
    # ------------------------------------------------------------------
    @property
    def provider(self) -> OCRProvider:
        """OCR provider, khởi tạo muộn để không tải model khi chỉ xử lý text."""
        if self._provider is None:
            self._provider = create_provider(
                self.engine, strict=self.strict_engine, **self.provider_kwargs
            )
        return self._provider

    def _extract_image_once(
        self, image_path: str, config: PreprocessConfig | None
    ) -> ExtractionResult:
        """Một lượt OCR + trích xuất với một cấu hình tiền xử lý cụ thể.

        Đổi thẳng ``preprocess_config`` của provider thay vì dựng provider mới:
        dựng lại sẽ nạp lại model recognition (và với PaddleX thì còn không dựng
        lại được - chỉ khởi tạo được một lần mỗi process).
        """
        warnings: list[str] = []
        provider = self.provider
        previous = getattr(provider, "preprocess_config", None)
        if config is not None and previous is not None:
            provider.preprocess_config = config
        try:
            ocr_result = provider.extract(image_path)
        except Exception as exc:
            message = f"OCR thất bại trên {image_path}: {exc}"
            logger.warning(message)
            warnings.append(message)
            ocr_result = OCRResult(tokens=[], image_path=image_path, provider="unavailable")
        finally:
            if config is not None and previous is not None:
                provider.preprocess_config = previous
        result = self.extract_from_ocr(ocr_result)
        result.warnings = warnings + result.warnings
        return result

    def extract_from_image(self, image_path: str) -> ExtractionResult:
        """Chạy toàn bộ pipeline trên một ảnh.

        Khi ``binarize`` là ``None`` (mặc định) thì chạy **cả hai** cấu hình tiền
        xử lý rồi giữ kết quả trích được nhiều field hơn - dùng đúng thang điểm
        của :func:`page_extraction_score`.

        Lý do không chọn sẵn một bên: nhị phân hoá **cần** cho Tesseract nhưng
        biến nét watermark mờ thành nét đen đặc đè lên chữ, còn detector DB của
        PaddleOCR thì ngược lại - bỏ nhị phân hoá chỉ làm nó bắt thêm token rác.
        Đo trên biểu mẫu BM.12 thật, hai engine cho kết quả **trái ngược nhau**,
        nên ép sẵn một bên là bắt người dùng phải đoán. Đánh đổi: chạy OCR hai
        lượt; ép bằng ``binarize=True/False`` để chỉ chạy một lượt.
        """
        if self.binarize is not None:
            config = PreprocessConfig(adaptive_threshold=self.binarize)
            return self._extract_image_once(image_path, config)

        # Provider không có tiền xử lý (mock trong test, hoặc provider tuỳ biến)
        # thì hai lượt cho kết quả y hệt - chạy một lượt là đủ.
        if not hasattr(self.provider, "preprocess_config"):
            return self._extract_image_once(image_path, None)

        attempts = [
            ("nhị phân hoá", PreprocessConfig(adaptive_threshold=True)),
            ("không nhị phân hoá", PreprocessConfig(adaptive_threshold=False)),
        ]
        best: tuple[tuple[int, float], str, ExtractionResult] | None = None
        for label, config in attempts:
            result = self._extract_image_once(image_path, config)
            score = page_extraction_score(result)
            logger.info("Tiền xử lý %s: trích được %d field", label, score[0])
            if best is None or score > best[0]:
                best = (score, label, result)
        assert best is not None
        logger.info("Chọn tiền xử lý '%s' (%d field)", best[1], best[0][0])
        return best[2]

    def extract_from_pdf(
        self,
        pdf_path: str,
        *,
        dpi: int = 300,
        pages: str | Sequence[int] | None = None,
        keep_pages: bool = False,
    ) -> ExtractionResult:
        """Tách PDF nhiều trang, OCR từng trang rồi gộp kết quả.

        :param dpi: độ phân giải render; 300 là mức đã kiểm chứng cả pipeline.
        :param pages: chuỗi kiểu ``"1,3-5"`` hoặc danh sách chỉ số 0-based;
            ``None`` = mọi trang.
        :param keep_pages: giữ lại thư mục ảnh tạm (soi khi debug).
        :raises PdfUnavailableError: thiếu ``pymupdf`` hoặc PDF không đọc được.
        """
        import shutil

        from ..pdf import make_page_dir, page_count, parse_page_selection, render_pages

        indices: Sequence[int] | None
        if pages is None or isinstance(pages, str):
            indices = parse_page_selection(pages, page_count(pdf_path))
        else:
            indices = list(pages)

        directory = make_page_dir()
        try:
            rendered = render_pages(pdf_path, directory, dpi=dpi, pages=indices)
            results = [
                (number, self.extract_from_image(image_path))
                for number, image_path in rendered
            ]
            for number, result in results:
                found = page_extraction_score(result)[0]
                logger.info("Trang %d: trích được %d field", number, found)
        finally:
            if keep_pages:
                logger.info("Ảnh từng trang được giữ ở %s", directory)
            else:
                shutil.rmtree(directory, ignore_errors=True)

        merged = merge_page_results(results)
        if merged.page is not None:
            logger.info(
                "Chọn trang %d làm trang chính trong %d trang", merged.page, len(results)
            )
        return merged

    def extract_from_ocr(self, ocr_result: OCRResult) -> ExtractionResult:
        """Dựng bảng từ kết quả OCR rồi trích xuất field."""
        build_result = build_table(
            ocr_result,
            use_morphology=self.use_morphology,
            use_pp_structure=self.use_pp_structure,
        )
        return self._extract(
            build_result,
            provider_name=ocr_result.provider,
            extra_warnings=list(ocr_result.warnings),
        )

    def extract_from_text(self, text: str) -> ExtractionResult:
        """Trích xuất từ một khối raw text (fixture test, hoặc OCR đã có sẵn)."""
        ocr_result = OCRResult.from_text(text, provider="text")
        build_result = build_table(ocr_result, use_morphology=False, use_pp_structure=False)
        return self._extract(build_result, provider_name="text")

    def extract_from_table(self, table: Table, raw_text: str = "") -> ExtractionResult:
        """Trích xuất từ một bảng đã dựng sẵn (test tầng extraction độc lập)."""
        build_result = TableBuildResult(table, table.source, raw_text or "", [])
        return self._extract(build_result, provider_name="table")

    # ------------------------------------------------------------------
    # Lõi trích xuất
    # ------------------------------------------------------------------
    def _extract(
        self,
        build_result: TableBuildResult,
        *,
        provider_name: str,
        extra_warnings: Iterable[str] = (),
    ) -> ExtractionResult:
        """Chạy resolve cho từng field trong schema."""
        warnings = list(extra_warnings) + list(build_result.warnings)
        results: dict[str, FieldResult] = {}
        for spec in self.schema.fields:
            field_result = self._resolve_field(spec, build_result)
            if not field_result.found:
                message = f"Không trích xuất được field {spec.name!r} - trả null"
                level = logger.warning if spec.required else logger.info
                level("%s", message)
                warnings.append(message)
            results[spec.name] = field_result

        warnings.extend(self._resolve_cell_collisions(results))

        return ExtractionResult(
            fields=results,
            raw_text=build_result.raw_text,
            table=build_result.table,
            table_strategy=build_result.strategy,
            ocr_provider=provider_name,
            warnings=warnings,
        )

    def _resolve_cell_collisions(self, results: dict[str, FieldResult]) -> list[str]:
        """Phát hiện 2 field khác nhau vô tình lấy giá trị từ cùng một ô bảng.

        Đây là lưới an toàn bổ sung cho việc alias fuzzy-match có thể khớp
        nhầm 2 field vào cùng một cột (VD alias của ``onnetMinutes`` từng
        khớp nhầm cột ``offnetMinutes``). Field có score thấp hơn bị đẩy về
        ``null`` kèm cảnh báo - thà thiếu còn hơn 2 field cùng báo sai một
        giá trị giống hệt nhau mà không ai biết.

        Field khai báo ``keyword`` trong ``parser_args`` (VD ``youtubeGB``/
        ``spotifyGB`` cùng đọc từ 1 ô gộp "Youtube: 25gb / Spotify: 25gb"
        nhưng mỗi field tự khoanh vùng theo keyword riêng) **được loại trừ**
        khỏi kiểm tra này - đó là thiết kế có chủ đích, không phải xung đột.
        """
        keyworded = {
            spec.name for spec in self.schema.fields if spec.parser_args.get("keyword")
        }
        by_cell: dict[tuple[int, int], list[FieldResult]] = {}
        for result in results.values():
            if result.source != SOURCE_TABLE or result.cell_coords is None:
                continue
            if result.name in keyworded:
                continue
            by_cell.setdefault(result.cell_coords, []).append(result)

        warnings: list[str] = []
        for coords, group in by_cell.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda r: r.score, reverse=True)
            winner, *losers = group
            for loser in losers:
                message = (
                    f"Field {loser.name!r} và {winner.name!r} cùng khớp ô {coords} "
                    f"(score {loser.score:.2f} vs {winner.score:.2f}) - giữ {winner.name!r}, "
                    f"trả null cho {loser.name!r}"
                )
                logger.warning(message)
                warnings.append(message)
                loser.value = None
                loser.source = SOURCE_MISSING
        return warnings

    def _resolve_field(self, spec: FieldSpec, build_result: TableBuildResult) -> FieldResult:
        """Tìm giá trị cho một field theo thứ tự: bảng -> dòng text -> regex."""
        candidates: list[tuple[str, str, float, str | None, tuple[int, int] | None]] = []

        if build_result.table is not None:
            for raw_value, score, alias, coords in self._find_in_table(spec, build_result.table):
                candidates.append((SOURCE_TABLE, raw_value, score, alias, coords))

        line_found = self._find_in_text_lines(spec, build_result.raw_text)
        if line_found is not None:
            raw_value, score, alias = line_found
            candidates.append((SOURCE_TEXT_LINE, raw_value, score, alias, None))

        regex_found = self._find_by_regex(spec, build_result.raw_text)
        if regex_found is not None:
            candidates.append((SOURCE_REGEX, regex_found, 0.7, None, None))

        for source, raw_value, score, alias, coords in candidates:
            value = parse_value(spec.parser, raw_value, **spec.parser_args)
            value = self._coerce_type(spec, value)
            if value is not None:
                return FieldResult(
                    name=spec.name,
                    value=value,
                    raw_value=raw_value,
                    source=source,
                    score=score,
                    matched_alias=alias,
                    cell_coords=coords if source == SOURCE_TABLE else None,
                )
            logger.debug(
                "Field %s: parser %s không parse được %r (source=%s)",
                spec.name,
                spec.parser,
                raw_value,
                source,
            )

        raw_value = candidates[0][1] if candidates else None
        return FieldResult(name=spec.name, value=None, raw_value=raw_value, source=SOURCE_MISSING)

    # ------------------------------------------------------------------
    # Chiến lược 1: tra trong bảng
    # ------------------------------------------------------------------
    def _find_in_table(
        self, spec: FieldSpec, table: Table
    ) -> list[tuple[str, float, str | None, tuple[int, int] | None]]:
        """Tìm ô nhãn khớp alias rồi liệt kê các ô giá trị ứng viên.

        Trả về danh sách ``(raw_value, score, alias)`` theo thứ tự ưu tiên. Việc
        chọn ứng viên nào là hợp lệ do parser quyết định (ứng viên nào parse ra
        ``None`` sẽ bị bỏ qua), nhờ đó bảng dọc (nhãn - giá trị cùng hàng) và bảng
        ngang (header trên, giá trị dưới) đều xử lý được mà không cần biết trước
        hướng của bảng.
        """
        settings = self.schema.settings
        best: tuple[float, str, str, int, int] | None = None  # (score, alias, text, row, col)

        for cell in table.iter_cells():
            if cell.is_empty:
                continue
            score, alias = match_alias(cell.text, spec.aliases, settings.min_similarity)
            if alias is None:
                continue
            if best is None or score > best[0]:
                best = (score, alias, cell.text, cell.row, cell.col)

        if best is None:
            return []

        score, alias, cell_text, row, col = best
        label_cell = table.cell_at(row, col)
        span_cols = label_cell.col_span if label_cell else 1
        span_rows = label_cell.row_span if label_cell else 1

        candidates: list[tuple[str, float, str | None, tuple[int, int] | None]] = []

        # Nhãn và giá trị nằm chung một ô: "Chu kỳ gia hạn: tháng".
        inline = remainder_after_alias(cell_text, alias)
        inline_candidate: tuple[str, float, str | None, tuple[int, int] | None] | None = (
            (inline, score, alias, (row, col)) if inline else None
        )

        # Ô bên phải (bảng dạng nhãn - giá trị) và ô bên dưới (bảng có header
        # trên cùng). Thứ tự ưu tiên phụ thuộc nhãn có nằm ở vùng header không.
        # Chỉ coi là "hàng header" khi bảng thực sự có >1 cột: với bảng 1 cột
        # dựng từ text thô (mỗi dòng = 1 ô), "ô bên dưới" luôn tồn tại một cách
        # tầm thường nên không phải tín hiệu đáng tin để suy ra đây là header.
        in_header = row < max(1, settings.header_rows) and table.n_cols > 1
        lookups = (
            (self._value_below, self._value_right)
            if in_header
            else (self._value_right, self._value_below)
        )
        adjacent: list[tuple[str, float, str | None, tuple[int, int] | None]] = []
        label_like: list[tuple[str, float, str | None, tuple[int, int] | None]] = []
        for lookup in lookups:
            found = lookup(table, row, col, span_rows, span_cols, settings.max_value_distance)
            if not found:
                continue
            value, value_row, value_col = found
            # Ô kế bên có thể lại là một nhãn khác (bảng nhãn - giá trị xếp dọc);
            # những ô như vậy bị đẩy xuống cuối danh sách ứng viên.
            if self._looks_like_label(value):
                label_like.append((value, score * 0.8, alias, (value_row, value_col)))
            else:
                adjacent.append((value, score, alias, (value_row, value_col)))

        if in_header:
            # Ở hàng header, giá trị gần như luôn nằm ở ô liền kề (dưới/phải);
            # phần dư ngay trong ô header (VD chú thích mã cột "(TK533)") chỉ
            # được thử SAU CÙNG, để tránh nó "thắng" ô giá trị thật.
            candidates.extend(adjacent)
            if inline_candidate:
                candidates.append(inline_candidate)
        else:
            if inline_candidate:
                candidates.append(inline_candidate)
            candidates.extend(adjacent)
        candidates.extend(label_like)

        if not candidates:
            logger.debug("Field %s: khớp nhãn %r nhưng không tìm thấy ô giá trị", spec.name, alias)
        return candidates

    def _looks_like_label(self, text: str) -> bool:
        """True nếu ô trông giống **nhãn** của một field nào đó chứ không phải giá trị.

        Ví dụ ``"Ưu đãi SMS"`` (khớp alias, không còn phần dư) là nhãn, còn
        ``"Youtube: 25gb"`` thì không vì phần sau nhãn vẫn có nội dung.
        """
        if not text:
            return False
        threshold = self.schema.settings.min_similarity
        for spec in self.schema.fields:
            _, alias = match_alias(text, spec.aliases, threshold)
            if alias is not None and not remainder_after_alias(text, alias):
                return True
        return False

    @staticmethod
    def _value_right(
        table: Table, row: int, col: int, row_span: int, col_span: int, max_distance: int
    ) -> tuple[str, int, int] | None:
        """Ô không rỗng đầu tiên nằm bên phải nhãn, trong cùng hàng (kèm toạ độ)."""
        start = col + col_span
        for offset in range(max(1, max_distance)):
            target = table.cell_at(row, start + offset)
            if target is None:
                continue
            if not target.is_empty:
                return target.text, target.row, target.col
        return None

    @staticmethod
    def _value_below(
        table: Table, row: int, col: int, row_span: int, col_span: int, max_distance: int
    ) -> tuple[str, int, int] | None:
        """Ô không rỗng đầu tiên nằm dưới nhãn, trong cùng cột (kèm toạ độ)."""
        start = row + row_span
        for offset in range(max(1, max_distance)):
            target = table.cell_at(start + offset, col)
            if target is None:
                continue
            if not target.is_empty:
                return target.text, target.row, target.col
        return None

    # ------------------------------------------------------------------
    # Chiến lược 2: tra theo từng dòng raw text
    # ------------------------------------------------------------------
    def _find_in_text_lines(self, spec: FieldSpec, raw_text: str) -> tuple[str, float, str | None] | None:
        """Tìm dòng bắt đầu bằng nhãn và lấy phần còn lại của dòng làm giá trị."""
        if not raw_text:
            return None
        settings = self.schema.settings
        best: tuple[float, str, str] | None = None  # (score, alias, value)

        # Nhãn của mọi field KHÁC đều là mốc kết thúc giá trị của field này.
        other_aliases = [
            alias
            for other in self.schema.fields
            if other.name != spec.name
            for alias in other.aliases
        ]

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            score, alias = match_alias(line, spec.aliases, settings.min_similarity)
            if alias is None:
                continue
            tail = remainder_after_alias(line, alias)
            value = cut_at_next_label(tail, spec.aliases, other_aliases)
            # Chỉ nhảy sang dòng kế tiếp khi sau nhãn KHÔNG còn gì. Nếu phần dư
            # có nội dung nhưng bị cắt cụt vì gặp nhãn của field khác thì giá
            # trị của field này thực sự vắng mặt - lấy dòng dưới là lấy nhầm
            # nội dung của cột khác.
            if not tail and index + 1 < len(lines):
                # Nhãn chiếm trọn dòng -> giá trị thường nằm ở dòng kế tiếp.
                candidate = lines[index + 1]
                next_score, _ = match_alias(candidate, spec.aliases, settings.min_similarity)
                if next_score < settings.min_similarity:
                    value = cut_at_next_label(candidate, spec.aliases, other_aliases)
            if not value:
                continue
            if best is None or score > best[0]:
                best = (score, alias, value)

        if best is None:
            return None
        score, alias, value = best
        return value, score, alias

    # ------------------------------------------------------------------
    # Chiến lược 3: regex trên raw text blob
    # ------------------------------------------------------------------
    @staticmethod
    def _find_by_regex(spec: FieldSpec, raw_text: str) -> str | None:
        """Dò các pattern fallback; khớp trên bản bỏ dấu nhưng trả text gốc còn dấu."""
        if not raw_text or not spec.regex:
            return None
        folded = fold_accents(raw_text)
        aligned = len(folded) == len(raw_text)
        if not aligned:  # pragma: no cover - fold_accents luôn giữ độ dài
            logger.debug("Bản bỏ dấu lệch độ dài, dùng trực tiếp bản bỏ dấu")

        for pattern in spec.regex:
            try:
                compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE)
            except re.error as exc:
                logger.warning("Field %s: regex %r không hợp lệ (%s)", spec.name, pattern, exc)
                continue
            for haystack, source_text in ((folded, raw_text if aligned else folded), (raw_text, raw_text)):
                match = compiled.search(haystack)
                if not match:
                    continue
                start, end = match.span(1) if match.groups() else match.span(0)
                value = source_text[start:end].strip()
                if value:
                    return value
        return None

    # ------------------------------------------------------------------
    # Ép kiểu theo khai báo schema
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_type(spec: FieldSpec, value: Any) -> Any:
        """Kiểm tra/ép giá trị về đúng kiểu khai báo; sai kiểu -> ``None`` + warning."""
        if value is None:
            return None
        declared = (spec.type or "string").lower()
        if declared in ("number", "float", "int", "integer"):
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return int(value) if declared in ("int", "integer") else value
            logger.warning("Field %s: giá trị %r không phải số", spec.name, value)
            return None
        if declared in ("array", "list"):
            if isinstance(value, list):
                return value
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            logger.warning("Field %s: giá trị %r không phải danh sách", spec.name, value)
            return None
        if isinstance(value, str):
            return value
        return str(value)


def extract_image_to_json(
    image_path: str,
    *,
    schema_path: str | Path | None = None,
    engine: str = "paddle_vietocr",
    use_morphology: bool = True,
    use_pp_structure: bool = True,
) -> dict[str, Any]:
    """Hàm tiện dụng: ảnh -> dict JSON (dùng cho script/notebook)."""
    extractor = CTKMExtractor(
        schema_path,
        engine=engine,
        use_morphology=use_morphology,
        use_pp_structure=use_pp_structure,
    )
    return extractor.extract_from_image(image_path).to_dict()
