"""Registry các hàm parser chuyển chuỗi OCR thô thành giá trị có kiểu.

Nguyên tắc chung của mọi parser:

* nhận đầu vào là chuỗi bất kỳ (kể cả rỗng / ``None`` / nhiễu OCR),
* trả về giá trị đã parse **hoặc** ``None`` nếu không suy ra được,
* **không bao giờ raise** - lỗi được ghi log ở mức warning.

Thêm parser mới chỉ cần khai báo hàm với decorator :func:`register` rồi trỏ tên
parser đó trong ``schema.yaml``; không phải sửa orchestrator.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

Parser = Callable[..., Any]

#: Registry tên parser -> hàm.
PARSERS: dict[str, Parser] = {}

#: Ký tự thường bị OCR nhầm khi nằm cạnh chữ số.
OCR_DIGIT_CONFUSIONS: dict[str, str] = {
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "l": "1",
    "I": "1",
    "|": "1",
    "!": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
    "b": "6",
}

#: Ký tự vừa là chữ số bị OCR nhầm, vừa là ĐƯỜNG KẺ DỌC của bảng.
#:
#: Chỉ đổi thành chữ số khi nằm GIỮA hai chữ số. Nằm ở đầu/cuối cụm thì gần như
#: luôn là đường kẻ ô bị OCR đọc lẫn vào nội dung: ``"163,636.3636|"`` mà đổi
#: thành ``163636.36361`` là bịa thêm một chữ số vào một số vốn đã đọc đúng.
OCR_BORDER_CONFUSIONS = frozenset("|!")

#: Ký tự đặc biệt tiếng Việt mà NFD không tách được dấu.
_SPECIAL_FOLD = {"đ": "d", "Đ": "D"}

#: Regex một cụm số có thể kèm phân cách hàng nghìn/thập phân.
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d\s.,]*\d|[-+]?\d")

_SEPARATORS = ".,"


def register(name: str) -> Callable[[Parser], Parser]:
    """Decorator đăng ký parser vào registry theo tên dùng trong ``schema.yaml``."""

    def decorator(func: Parser) -> Parser:
        if name in PARSERS:
            logger.warning("Parser %r bị đăng ký trùng, ghi đè bản cũ", name)
        PARSERS[name] = func
        return func

    return decorator


def get_parser(name: str) -> Parser:
    """Lấy parser theo tên; nếu không có thì cảnh báo và dùng ``text_parser``."""
    parser = PARSERS.get(name)
    if parser is None:
        logger.warning("Không tìm thấy parser %r, dùng text_parser thay thế", name)
        return PARSERS["text_parser"]
    return parser


def parse_value(parser_name: str, raw: Any, **kwargs: Any) -> Any:
    """Gọi parser an toàn: mọi exception đều thành ``None`` + log warning."""
    parser = get_parser(parser_name)
    try:
        return parser(raw, **kwargs)
    except Exception as exc:  # pragma: no cover - lưới an toàn cuối cùng
        logger.warning("Parser %r lỗi với giá trị %r: %s", parser_name, raw, exc)
        return None


# ---------------------------------------------------------------------------
# Tiện ích chuẩn hoá chuỗi
# ---------------------------------------------------------------------------
def fold_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt **giữ nguyên độ dài chuỗi** (map 1 ký tự -> 1 ký tự).

    Giữ nguyên độ dài để có thể dò regex trên bản không dấu rồi cắt giá trị từ
    chuỗi gốc (còn dấu) theo đúng vị trí.
    """
    result: list[str] = []
    for char in text or "":
        if char in _SPECIAL_FOLD:
            result.append(_SPECIAL_FOLD[char])
            continue
        decomposed = unicodedata.normalize("NFD", char)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        result.append(stripped[0] if stripped else char)
    return "".join(result)


def normalize_label(text: str) -> str:
    """Chuẩn hoá nhãn để so khớp header: bỏ dấu, lowercase, gom khoảng trắng."""
    folded = fold_accents(text or "").lower()
    folded = re.sub(r"[^a-z0-9+]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def coerce_text(raw: Any) -> str:
    """Ép đầu vào về chuỗi đã strip; kiểu không hỗ trợ -> chuỗi rỗng.

    Cố tình **không** dùng ``str(raw)`` cho object bất kỳ: repr của object chứa địa
    chỉ hex và sẽ bị hiểu nhầm thành số.
    """
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, (list, tuple)):
        return " ".join(coerce_text(item) for item in raw).strip()
    logger.debug("coerce_text: bỏ qua giá trị kiểu %s", type(raw).__name__)
    return ""


def fix_ocr_digits(text: str) -> str:
    """Sửa ký tự bị OCR nhầm **chỉ khi** nó nằm cạnh một chữ số.

    Hai mức xử lý, đều bám vào "ngữ cảnh số" để không phá hỏng chữ:

    1. Cụm (tách theo khoảng trắng) chỉ gồm chữ số, dấu phân cách và ký tự dễ nhầm
       -> sửa toàn bộ: ``"15O.OOO"`` -> ``"150.000"``.
    2. Các trường hợp còn lại chỉ sửa ký tự **liền kề** một chữ số:
       ``"1O0 SMS"`` -> ``"100 SMS"``, còn ``"150 sms"`` và ``"8gb"`` giữ nguyên.

    Riêng ``|`` và ``!`` (xem :data:`OCR_BORDER_CONFUSIONS`) chỉ được sửa khi nằm
    GIỮA hai chữ số, vì ở rìa cụm chúng thường là đường kẻ ô của bảng.
    """
    if not text:
        return ""

    def convertible(chunk: str, index: int) -> bool:
        """``chunk[index]`` có được phép đổi thành chữ số không?"""
        char = chunk[index]
        if char not in OCR_DIGIT_CONFUSIONS:
            return False
        previous = chunk[index - 1] if index > 0 else ""
        following = chunk[index + 1] if index + 1 < len(chunk) else ""
        if char in OCR_BORDER_CONFUSIONS:
            # Phải nằm TRONG LÒNG cụm số (hai bên đều là chữ số hoặc dấu phân
            # cách) mới là chữ số; ở rìa cụm thì là đường kẻ ô.
            # Chú ý: phải kiểm tra chuỗi rỗng trước, vì "" in ".," là True.
            inside = bool(previous and following) and (
                previous.isdigit() or previous in _SEPARATORS
            ) and (following.isdigit() or following in _SEPARATORS)
            return inside and (previous.isdigit() or following.isdigit())
        return previous.isdigit() or following.isdigit()

    def fix_chunk(chunk: str) -> str:
        if not chunk:
            return chunk
        chars = list(chunk)
        if any(char.isdigit() for char in chunk) and all(
            char.isdigit() or char in _SEPARATORS or char in OCR_DIGIT_CONFUSIONS
            for char in chunk
        ):
            # Mức 1: cụm thuần số -> sửa hết, trừ đường kẻ ở rìa.
            for index, char in enumerate(chunk):
                if char in OCR_BORDER_CONFUSIONS and not convertible(chunk, index):
                    continue
                chars[index] = OCR_DIGIT_CONFUSIONS.get(char, char)
            return "".join(chars)
        # Mức 2: chỉ sửa ký tự liền kề một chữ số.
        for index in range(len(chunk)):
            if convertible(chunk, index):
                chars[index] = OCR_DIGIT_CONFUSIONS[chunk[index]]
        return "".join(chars)

    # re.split giữ lại phần khoảng trắng nên độ dài chuỗi không đổi.
    return "".join(
        part if part.isspace() else fix_chunk(part) for part in re.split(r"(\s+)", text)
    )


def slice_after_keyword(text: str, keyword: str | None) -> str:
    """Phần chuỗi nằm sau ``keyword`` (so khớp không dấu, không phân biệt hoa thường).

    Trả về chuỗi gốc nếu không có keyword hoặc không tìm thấy.
    """
    if not keyword:
        return text
    folded_text = fold_accents(text or "").lower()
    folded_keyword = fold_accents(keyword).lower()
    position = folded_text.find(folded_keyword)
    if position < 0:
        return text
    return text[position + len(folded_keyword) :]


# ---------------------------------------------------------------------------
# Parser số
# ---------------------------------------------------------------------------
def _split_number_string(cleaned: str) -> tuple[str, str]:
    """Tách chuỗi số thành ``(phần nguyên, phần thập phân)``.

    Heuristic xử lý chuỗi lẫn lộn ``,`` và ``.``:

    * Có cả hai loại dấu -> dấu **xuất hiện sau cùng** là dấu thập phân
      (``"163,636.3636"`` -> ``163636.3636``).
    * Chỉ một loại, xuất hiện nhiều lần và mọi nhóm phía sau đều đúng 3 chữ số ->
      toàn bộ là phân cách hàng nghìn (``"150.534.213"`` -> ``150534213``).
    * Chỉ một loại, xuất hiện một lần -> là hàng nghìn nếu nhóm sau đúng 3 chữ số,
      ngược lại là dấu thập phân (``"163,6363"`` -> ``163.6363``).
    """
    positions = [index for index, char in enumerate(cleaned) if char in _SEPARATORS]
    if not positions:
        return cleaned, ""

    groups = []
    for order, position in enumerate(positions):
        end = positions[order + 1] if order + 1 < len(positions) else len(cleaned)
        groups.append(cleaned[position + 1 : end])

    distinct = {cleaned[position] for position in positions}
    decimal_position: int | None
    if len(distinct) > 1:
        decimal_position = positions[-1]
    elif len(positions) > 1:
        decimal_position = None if all(len(group) == 3 for group in groups) else positions[-1]
    else:
        decimal_position = None if len(groups[0]) == 3 else positions[0]

    if decimal_position is None:
        return re.sub(r"[.,]", "", cleaned), ""
    integer_part = re.sub(r"[.,]", "", cleaned[:decimal_position])
    fraction_part = re.sub(r"[^0-9]", "", cleaned[decimal_position + 1 :])
    return integer_part, fraction_part


def _to_number(cleaned: str) -> float | int | None:
    """Chuyển chuỗi số đã làm sạch thành ``int`` hoặc ``float``."""
    sign = -1 if cleaned.startswith("-") else 1
    cleaned = cleaned.lstrip("+-").strip()
    cleaned = cleaned.strip(_SEPARATORS)
    if not cleaned or not any(char.isdigit() for char in cleaned):
        return None
    integer_part, fraction_part = _split_number_string(cleaned)
    integer_part = integer_part or "0"
    if not fraction_part or set(fraction_part) == {"0"}:
        try:
            return sign * int(integer_part)
        except ValueError:
            return None
    try:
        return sign * float(f"{integer_part}.{fraction_part}")
    except ValueError:
        return None


def _find_number_string(text: str, occurrence: int = 0) -> str | None:
    """Lấy cụm số thứ ``occurrence`` trong chuỗi (đã bỏ khoảng trắng bên trong)."""
    matches = NUMBER_PATTERN.findall(text)
    if not matches or occurrence >= len(matches):
        return None
    return re.sub(r"\s+", "", matches[occurrence])


@register("money_parser")
def money_parser(
    raw: Any,
    *,
    keyword: str | None = None,
    occurrence: int = 0,
    fix_ocr: bool = True,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | int | None:
    """Parse số tiền có thể lẫn lộn dấu ``,`` và ``.``.

    Ví dụ: ``"163,636.3636 đ"`` -> ``163636.3636``; ``"150.534.213"`` -> ``150534213``;
    ``"Miễn phí"`` -> ``None``.
    """
    text = coerce_text(raw)
    if not text:
        return None
    text = slice_after_keyword(text, keyword)
    if fix_ocr:
        text = fix_ocr_digits(text)
    number_string = _find_number_string(text, occurrence)
    if number_string is None:
        logger.debug("money_parser không tìm thấy số trong %r", text)
        return None
    value = _to_number(number_string)
    if value is None:
        return None
    if min_value is not None and value < min_value:
        logger.warning("money_parser: giá trị %s nhỏ hơn ngưỡng %s", value, min_value)
        return None
    if max_value is not None and value > max_value:
        logger.warning("money_parser: giá trị %s lớn hơn ngưỡng %s", value, max_value)
        return None
    return value


@register("int_parser")
def int_parser(
    raw: Any,
    *,
    keyword: str | None = None,
    occurrence: int = 0,
    fix_ocr: bool = True,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Lấy số nguyên đầu tiên trong chuỗi.

    Ví dụ: ``"150"`` -> ``150``; ``"MP 20p đầu tiên"`` -> ``20``;
    ``"1.500 phút"`` -> ``1500``; ``""`` -> ``None``.
    """
    value = money_parser(raw, keyword=keyword, occurrence=occurrence, fix_ocr=fix_ocr)
    if value is None:
        return None
    try:
        result = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        logger.warning("int_parser không ép được %r về số nguyên", value)
        return None
    if min_value is not None and result < min_value:
        logger.warning("int_parser: giá trị %s nhỏ hơn ngưỡng %s", result, min_value)
        return None
    if max_value is not None and result > max_value:
        logger.warning("int_parser: giá trị %s lớn hơn ngưỡng %s", result, max_value)
        return None
    return result


@register("gb_parser")
def gb_parser(
    raw: Any,
    *,
    keyword: str | None = None,
    unit_pattern: str = r"g\s?b",
    fallback_to_number: bool = True,
    require_keyword: bool = False,
    fix_ocr: bool = True,
) -> float | int | None:
    """Lấy dung lượng GB đầu tiên trong chuỗi.

    Ví dụ: ``"60GB/tháng, tối đa 8gb/1 ngày"`` -> ``60``;
    ``"Youtube: 25gb/tháng"`` (keyword ``"Youtube"``) -> ``25``.

    ``keyword`` chỉ dùng để **thu hẹp** phạm vi tìm kiếm: nếu không thấy keyword thì
    vẫn dò trên toàn chuỗi (vì phạm vi đã được giới hạn bởi ô bảng hoặc nhóm regex),
    trừ khi bật ``require_keyword=True``.
    ``fallback_to_number=True`` cho phép lấy số đầu tiên khi ô chỉ ghi ``"60"`` vì
    đơn vị đã nằm ở header cột.
    """
    text = coerce_text(raw)
    if not text:
        return None
    if keyword and require_keyword and not _contains_keyword(text, keyword):
        logger.debug("gb_parser: không thấy keyword %r trong %r", keyword, text)
        return None
    scoped = slice_after_keyword(text, keyword)
    if fix_ocr:
        scoped = fix_ocr_digits(scoped)
    pattern = re.compile(rf"([-+]?\d[\d\s.,]*\d|\d)\s*{unit_pattern}", re.IGNORECASE)
    match = pattern.search(scoped)
    if match:
        return _to_number(re.sub(r"\s+", "", match.group(1)))
    if fallback_to_number:
        number_string = _find_number_string(scoped)
        if number_string is not None:
            return _to_number(number_string)
    logger.debug("gb_parser không tìm thấy dung lượng trong %r", text)
    return None


def _contains_keyword(text: str, keyword: str) -> bool:
    """True nếu ``keyword`` xuất hiện trong ``text`` (bỏ dấu, không phân biệt hoa thường)."""
    return fold_accents(keyword).lower() in fold_accents(text).lower()


# ---------------------------------------------------------------------------
# Parser chuỗi
# ---------------------------------------------------------------------------
@register("text_parser")
def text_parser(
    raw: Any,
    *,
    pattern: str | None = None,
    upper: bool = False,
    lower: bool = False,
    strip_chars: str = " \t\r\n:;.-–—|",
    max_length: int | None = None,
    keyword: str | None = None,
) -> str | None:
    """Trả chuỗi đã làm sạch; có thể lọc theo ``pattern`` (nhóm 1 hoặc toàn khớp)."""
    text = coerce_text(raw)
    if not text:
        return None
    text = slice_after_keyword(text, keyword).strip(strip_chars).strip()
    if pattern:
        try:
            match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        except re.error as exc:
            logger.warning("text_parser: pattern %r không hợp lệ (%s)", pattern, exc)
            match = None
        if not match:
            logger.debug("text_parser: %r không khớp pattern %r", text, pattern)
            return None
        text = match.group(1) if match.groups() else match.group(0)
    text = text.strip(strip_chars).strip()
    if not text:
        return None
    if max_length is not None and len(text) > max_length:
        text = text[:max_length].strip()
    if upper:
        text = text.upper()
    elif lower:
        text = text.lower()
    return text or None


@register("cycle_parser")
def cycle_parser(
    raw: Any,
    *,
    keywords: Sequence[str] | None = None,
    prefer_after: Sequence[str] | None = None,
    lower: bool = True,
) -> str | None:
    """Trích từ khoá chu kỳ (``tháng``/``ngày``/``tuần``...) trong chuỗi.

    Ví dụ: ``"Chu kỳ gia hạn: tháng"`` -> ``"tháng"``;
    ``"30 ngày"`` -> ``"ngày"``.
    """
    text = coerce_text(raw)
    if not text:
        return None
    candidates = list(keywords or ("tháng", "ngày", "tuần", "quý", "năm"))
    prefixes = list(prefer_after or ("chu kỳ", "chu ky", "cycle", "gia hạn", "gia han"))

    scoped = text
    folded_full = fold_accents(text).lower()
    for prefix in prefixes:
        position = folded_full.find(fold_accents(prefix).lower())
        if position >= 0:
            scoped = text[position + len(prefix) :]
            break

    for source in (scoped, text):
        folded = fold_accents(source).lower()
        best_position = -1
        best_keyword: str | None = None
        for candidate in candidates:
            position = folded.find(fold_accents(candidate).lower())
            if position >= 0 and (best_position < 0 or position < best_position):
                best_position = position
                best_keyword = candidate
        if best_keyword is not None:
            return best_keyword.lower() if lower else best_keyword
    logger.debug("cycle_parser không tìm thấy từ khoá chu kỳ trong %r", text)
    return None


@register("list_parser")
def list_parser(
    raw: Any,
    *,
    separators: Sequence[str] | None = None,
    strip_chars: str = " \t\r\n.;:|-–—",
    min_item_length: int = 1,
    max_items: int | None = None,
    keyword: str | None = None,
    unique: bool = True,
) -> list[str] | None:
    """Tách danh sách theo dấu phẩy (hoặc dấu khác) thành list chuỗi.

    Ví dụ: ``"Basic+, Family, Corporate++"`` ->
    ``["Basic+", "Family", "Corporate++"]``. Trả ``None`` khi không còn phần tử nào.
    """
    text = coerce_text(raw)
    if not text:
        return None
    text = slice_after_keyword(text, keyword)
    tokens = [sep for sep in (separators or [",", ";", "\n", "/", "|"])]
    pattern = "|".join(re.escape(sep) for sep in tokens if sep)
    parts = re.split(pattern, text) if pattern else [text]

    items: list[str] = []
    for part in parts:
        cleaned = part.strip().strip(strip_chars).strip()
        if len(cleaned) < max(1, min_item_length):
            continue
        if unique and cleaned in items:
            continue
        items.append(cleaned)
        if max_items is not None and len(items) >= max_items:
            break
    if not items:
        logger.debug("list_parser không tách được phần tử nào từ %r", text)
        return None
    return items


@register("raw_parser")
def raw_parser(raw: Any) -> str | None:
    """Trả chuỗi gần như nguyên trạng (chỉ strip) - dùng khi cần giữ định dạng gốc."""
    text = coerce_text(raw)
    return text or None


def available_parsers() -> list[str]:
    """Danh sách tên parser đang đăng ký (tiện cho log/debug)."""
    return sorted(PARSERS)


def parse_many(specs: Iterable[tuple[str, str, Any, dict[str, Any]]]) -> dict[str, Any]:
    """Chạy nhiều parser một lượt.

    ``specs`` là chuỗi tuple ``(field_name, parser_name, raw_value, kwargs)``.
    """
    return {
        field_name: parse_value(parser_name, raw, **dict(kwargs or {}))
        for field_name, parser_name, raw, kwargs in specs
    }
