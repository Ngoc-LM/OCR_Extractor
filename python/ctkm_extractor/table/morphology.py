"""Dựng cấu trúc bảng bằng CV cổ điển: morphology tách đường kẻ + connected
components lấy lưới ô, thay vì dùng model học sâu (PP-Structure).

Lý do ưu tiên cách này cho CTKM: bảng trong phiếu có đường kẻ rõ (bordered
table). Với loại bảng này, tách đường kẻ bằng morphological opening là
tất định (deterministic), không cần tải thêm model, và không phụ thuộc
``paddleocr`` - nên vẫn hoạt động được kể cả khi chỉ dùng OCR provider dự
phòng (Tesseract). PP-Structure vẫn được giữ làm lớp dự phòng tiếp theo cho
bảng không viền hoặc layout bất thường mà cách này không dò được đường kẻ.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..ocr.base import BoundingBox, OCRToken
from .reconstruct import Table, TableCell, detect_column_bands, join_tokens

logger = logging.getLogger(__name__)

#: Cần tối thiểu chừng này đường kẻ ngang/dọc mới coi là "có bảng viền rõ".
MIN_LINES = 3

#: Độ dài tối thiểu của một đoạn được coi là đường kẻ bảng, tính theo tỉ lệ
#: cạnh tương ứng của ảnh.
#:
#: LƯU Ý - vì sao 0.05 chứ không phải 0.35 như bản đầu: kernel dài bằng 0.35
#: cạnh ảnh chỉ giữ được đường kẻ của bảng LỚN NHẤT trang. Trên biểu mẫu BM.12
#: thật (2480x3505), bảng CTKM nằm lồng trong một ô của bảng ngoài và chỉ cao
#: ~410px, nên toàn bộ 9 đường kẻ dọc của nó bị phép opening xoá sạch (kernel
#: yêu cầu 1226px liên tục) - dò ra đúng 5 đường dọc của bảng ngoài. Hạ xuống
#: 0.05 thì dò đủ 14 đường dọc (5 ngoài + 9 trong). Nét chữ không sống sót nổi
#: một kernel dài hàng trăm pixel nên không sinh đường kẻ giả.
MIN_LINE_LENGTH_RATIO = 0.05

#: Dung sai (pixel) khi xét một đường kẻ dọc có cắt hết một dải hàng hay không.
BAND_COVERAGE_TOLERANCE = 5

#: Khoảng cách (pixel) giữa 2 vị trí đường kẻ được coi là cùng một đường
#: (gộp lại tránh đường kẻ dày/nén JPEG/khử răng cưa bị đếm thành nhiều
#: đường sát nhau - kiểm chứng thực nghiệm trên ảnh scan 200 DPI thật).
LINE_MERGE_GAP = 15


class TableMorphologyUnavailableError(RuntimeError):
    """Thiếu OpenCV, không đọc được ảnh, hoặc không dò đủ đường kẻ bảng."""


def _load_gray(cv2: Any, np: Any, image: Any):
    """Nạp ảnh thành ma trận grayscale; chấp nhận đường dẫn hoặc mảng numpy."""
    if isinstance(image, str):
        mat = cv2.imread(image)
        if mat is None:
            raise TableMorphologyUnavailableError(f"Không đọc được ảnh: {image}")
    elif isinstance(image, np.ndarray):
        mat = image
    else:
        raise TableMorphologyUnavailableError(f"Kiểu ảnh không hỗ trợ: {type(image)!r}")
    if mat.ndim == 3:
        return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    return mat


def _binarize(cv2: Any, gray: Any):
    """Nhị phân hoá thích nghi: chữ/đường kẻ tối -> trắng (255) trên nền đen."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15,
    )


def _extract_lines(cv2: Any, binary: Any) -> tuple[Any, Any]:
    """Tách mặt nạ đường kẻ ngang và dọc bằng morphological opening.

    Ý tưởng: 1 kernel hình chữ nhật RẤT dài theo 1 trục sẽ chỉ "sống sót" qua
    phép opening nếu tồn tại 1 đoạn liên tục đủ dài theo trục đó trong ảnh nhị
    phân - đúng đặc điểm của đường kẻ bảng, khác với nét chữ (ngắn, đứt đoạn).
    """
    h, w = binary.shape[:2]
    h_len = max(15, int(w * MIN_LINE_LENGTH_RATIO))
    v_len = max(15, int(h * MIN_LINE_LENGTH_RATIO))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    return horizontal, vertical


def _line_positions(mask: Any, axis: int) -> list[float]:
    """Toạ độ trung tâm của từng đường kẻ, suy ra từ hình chiếu mật độ pixel.

    ``axis=1`` chiếu theo hàng (tìm đường NGANG, mỗi đường ứng với 1 giá trị y);
    ``axis=0`` chiếu theo cột (tìm đường DỌC, mỗi đường ứng với 1 giá trị x).
    """
    projection = mask.sum(axis=axis)
    peak = projection.max()
    if peak <= 0:
        return []
    threshold = peak * 0.3
    positions = [index for index, value in enumerate(projection) if value > threshold]
    if not positions:
        return []
    groups: list[list[int]] = [[positions[0]]]
    for position in positions[1:]:
        if position - groups[-1][-1] <= LINE_MERGE_GAP:
            groups[-1].append(position)
        else:
            groups.append([position])
    return [sum(group) / len(group) for group in groups]


def detect_grid(image: Any) -> tuple[list[float], list[float]]:
    """Dò toạ độ các đường kẻ ngang/dọc từ ảnh; ném lỗi nếu OpenCV thiếu.

    Hàm tách riêng khỏi :class:`MorphologyTableRecognizer` để tiện unit-test
    phần dò lưới độc lập với việc gán token.
    """
    row_lines, col_segments = detect_grid_segments(image)
    return row_lines, [segment[0] for segment in col_segments]


def detect_grid_segments(image: Any) -> tuple[list[float], list[tuple[float, float, float]]]:
    """Như :func:`detect_grid` nhưng đường dọc kèm phạm vi ``(x, y_đầu, y_cuối)``.

    Phạm vi y của đường kẻ dọc là thứ phân biệt bảng lồng nhau: đường kẻ của
    bảng ngoài trải dài toàn bảng, còn đường kẻ của bảng con chỉ nằm trong một
    dải hàng.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise TableMorphologyUnavailableError(f"Thiếu opencv-python/numpy: {exc}") from exc

    gray = _load_gray(cv2, np, image)
    binary = _binarize(cv2, gray)
    horizontal_mask, vertical_mask = _extract_lines(cv2, binary)
    row_lines = sorted(_line_positions(horizontal_mask, axis=1))

    segments: list[tuple[float, float, float]] = []
    for x in sorted(_line_positions(vertical_mask, axis=0)):
        column = int(round(x))
        window = vertical_mask[:, max(0, column - 3) : column + 4]
        rows_on = np.where(window.max(axis=1) > 0)[0]
        if len(rows_on):
            segments.append((x, float(rows_on.min()), float(rows_on.max())))
    return row_lines, segments


def columns_in_band(
    col_segments: Sequence[tuple[float, float, float]], top: float, bottom: float
) -> list[float]:
    """Toạ độ x của các đường kẻ dọc cắt HẾT dải hàng ``[top, bottom]``.

    Đây là biên cột chính xác nhất khi bảng có kẻ dọc rõ - chính xác hơn suy từ
    toạ độ token, vì 2 tiêu đề cột nằm sát nhau dễ bị gộp thành một band.
    """
    # Chỉ giữ đường kẻ THUỘC VỀ dải này: đường của bảng bao ngoài cũng cắt qua
    # dải, nhưng kéo dài vượt xa hai đầu nên bị loại bằng biên độ dưới đây.
    margin = max(BAND_COVERAGE_TOLERANCE, (bottom - top) * 0.5)
    return sorted(
        x
        for x, y_start, y_end in col_segments
        if y_start <= top + BAND_COVERAGE_TOLERANCE
        and y_end >= bottom - BAND_COVERAGE_TOLERANCE
        and y_start >= top - margin
        and y_end <= bottom + margin
    )


def select_densest_band(
    row_lines: Sequence[float], col_segments: Sequence[tuple[float, float, float]]
) -> tuple[float, float] | None:
    """Chọn dải hàng có NHIỀU ĐƯỜNG KẺ DỌC CẮT QUA NHẤT - tức bảng "dày" nhất.

    Dùng để tách bảng con ra khỏi bảng lồng nhau: trên biểu mẫu BM.12 thật, mọi
    dải hàng của bảng ngoài chỉ có 5 đường dọc cắt qua, riêng 2 dải chứa bảng
    CTKM có 14 - chọn đúng 2 dải đó là ra bảng cần trích xuất.

    Trả ``None`` khi mọi dải đều có số đường dọc như nhau (ảnh chỉ có một bảng),
    khi đó tầng gọi giữ nguyên toàn bộ lưới.
    """
    if len(row_lines) < 2 or not col_segments:
        return None

    counts: list[int] = []
    for index in range(len(row_lines) - 1):
        top, bottom = row_lines[index], row_lines[index + 1]
        counts.append(
            sum(
                1
                for _, y_start, y_end in col_segments
                if y_start <= top + BAND_COVERAGE_TOLERANCE
                and y_end >= bottom - BAND_COVERAGE_TOLERANCE
            )
        )

    best = max(counts)
    if best == min(counts):
        return None  # lưới đồng nhất - chỉ có một bảng

    # Dải liên tiếp dài nhất đạt mức dày nhất.
    best_start = best_length = current_start = current_length = 0
    for index, count in enumerate(counts):
        if count == best:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_start, best_length = current_start, current_length
        else:
            current_length = 0
    return row_lines[best_start], row_lines[best_start + best_length]


class MorphologyTableRecognizer:
    """Nhận diện cấu trúc bảng bằng morphology cổ điển - không cần model."""

    name = "morphology"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            return False
        return True

    def recognize(self, image: Any, tokens: Sequence[OCRToken]) -> Table:
        """Dò HÀNG từ đường kẻ ngang (chính xác kể cả ô gộp nhiều dòng - không
        có đường kẻ ngang giữa 2 dòng chữ được wrap trong cùng 1 ô), dò CỘT từ
        đường kẻ dọc của dải hàng đang xét và **chỉ** suy từ toạ độ token hàng
        header khi bảng không có đủ kẻ dọc, rồi gán token vào lưới
        (hàng, cột) kết quả.

        :raises TableMorphologyUnavailableError: khi thiếu OpenCV, không đọc
            được ảnh, không đủ đường kẻ ngang để coi là bảng viền rõ, hoặc
            không tách được cột nào từ token.
        """
        row_lines, col_segments = detect_grid_segments(image)
        if len(row_lines) < MIN_LINES:
            raise TableMorphologyUnavailableError(
                f"Không đủ đường kẻ ngang để coi là bảng viền rõ "
                f"(hàng={len(row_lines)}, cần >= {MIN_LINES})"
            )

        usable = [t for t in tokens if not t.is_empty]

        # BẢNG LỒNG BẢNG: giữ lại đúng dải hàng có nhiều đường kẻ dọc cắt qua
        # nhất. Nếu không làm bước này, hàng 0 của lưới là header của bảng
        # NGOÀI, nên biên cột suy từ token hàng 0 sẽ là cột của bảng ngoài chứ
        # không phải bảng dữ liệu bên trong.
        band = select_densest_band(row_lines, col_segments)
        if band is not None:
            top, bottom = band
            selected = [line for line in row_lines if top - 1 <= line <= bottom + 1]
            inside = [
                token
                for token in usable
                if top <= token.box.center[1] <= bottom
            ]
            if len(selected) >= MIN_LINES and inside:
                logger.info(
                    "Phát hiện bảng lồng nhau: thu hẹp về dải y %.0f-%.0f "
                    "(%d/%d đường ngang, %d/%d token)",
                    top,
                    bottom,
                    len(selected),
                    len(row_lines),
                    len(inside),
                    len(usable),
                )
                row_lines, usable = selected, inside
        trailing_limit = _trailing_row_limit(row_lines)
        row_of: dict[int, int | None] = {}
        for index, token in enumerate(usable):
            row_of[index] = _row_for_token(token.box, row_lines, trailing_limit)

        # Dò biên cột chỉ từ token ở HÀNG HEADER (hàng 0), không gộp toàn bộ
        # token mọi hàng: token ở hàng dữ liệu có thể rộng hơn cột chứa nó
        # (VD "MP 20p đầu tiên" tràn ra ngoài cột hẹp "TK thoại"), làm 2 band
        # kề nhau bị nối nhầm thành 1 nếu tính gộp. Thực nghiệm trên ảnh scan
        # thật: gộp toàn bộ token ra 6/8 cột (sai), chỉ dùng hàng header ra
        # đúng 8/8 cột.
        # Ưu tiên ĐƯỜNG KẺ DỌC thật làm biên cột; chỉ suy từ token khi bảng
        # không có kẻ dọc. Trên BM.12, biên suy từ token gộp nhầm "Phí đăng ký"
        # với "TK thoại" thành một cột, còn đường kẻ dọc cho đúng 8 cột.
        vertical = columns_in_band(col_segments, row_lines[0], row_lines[-1])
        if len(vertical) >= MIN_LINES:
            bands = [(vertical[i], vertical[i + 1]) for i in range(len(vertical) - 1)]
            # Loại token nằm ngoài bề ngang của bảng (VD chữ của bảng bao ngoài).
            usable = [
                token
                for token in usable
                if vertical[0] - BAND_COVERAGE_TOLERANCE
                <= token.box.center[0]
                <= vertical[-1] + BAND_COVERAGE_TOLERANCE
            ]
            row_of = {
                index: _row_for_token(token.box, row_lines, trailing_limit)
                for index, token in enumerate(usable)
            }
        else:
            header_tokens = [t for i, t in enumerate(usable) if row_of[i] == 0]
            bands = detect_column_bands(header_tokens or usable)
        if not bands:
            raise TableMorphologyUnavailableError("Không tách được cột nào từ token OCR")

        grouped: dict[tuple[int, int], list[OCRToken]] = {}
        unmatched = 0
        for index, token in enumerate(usable):
            row_index = row_of[index]
            if row_index is None:
                unmatched += 1
                continue
            col_index = _best_band(token.box, bands)
            grouped.setdefault((row_index, col_index), []).append(token)

        if not grouped:
            raise TableMorphologyUnavailableError(
                "Không gán được token nào vào lưới (hàng, cột) đã dò"
            )

        cells: list[TableCell] = []
        for (row_index, col_index), cell_tokens in sorted(grouped.items()):
            box = cell_tokens[0].box
            for token in cell_tokens[1:]:
                box = box.merge(token.box)
            cells.append(
                TableCell(
                    row=row_index,
                    col=col_index,
                    box=box,
                    text=join_tokens(cell_tokens),
                    tokens=list(cell_tokens),
                )
            )

        table = Table(
            cells=cells,
            n_rows=len(row_lines) - 1,
            n_cols=len(bands),
            source="morphology",
            confidence=0.8,
        )
        logger.info(
            "Dựng bảng bằng morphology (hàng theo đường kẻ, cột theo token): "
            "%d hàng x %d cột, gán %d/%d token (%d token không khớp hàng nào)",
            table.n_rows,
            table.n_cols,
            sum(len(v) for v in grouped.values()),
            len(usable),
            unmatched,
        )
        return table


def _row_for_token(
    box: BoundingBox, row_lines: Sequence[float], trailing_limit: float | None = None
) -> int | None:
    """Chỉ số hàng chứa tâm của ``box``, suy từ các đường kẻ ngang đã dò.

    ``trailing_limit`` (nếu có) là toạ độ y tối đa còn được coi là thuộc **hàng
    cuối chưa đóng**: ảnh scan/cắt cúp thường thiếu đường kẻ dưới cùng, khi đó
    token của hàng cuối nằm dưới đường kẻ cuối và sẽ bị mất trắng nếu không có
    ngoại lệ này. Token nằm phía TRÊN đường kẻ đầu (tiêu đề trang, số hiệu biểu
    mẫu) vẫn bị loại vì không thuộc bảng.
    """
    center_y = box.center[1]
    for index in range(len(row_lines) - 1):
        if row_lines[index] <= center_y < row_lines[index + 1]:
            return index
    if (
        trailing_limit is not None
        and row_lines
        and row_lines[-1] <= center_y <= trailing_limit
    ):
        return len(row_lines) - 1
    return None


def _trailing_row_limit(row_lines: Sequence[float]) -> float | None:
    """Giới hạn y của "hàng cuối chưa đóng", bằng 1 bước hàng kể từ đường kẻ cuối."""
    if len(row_lines) < 2:
        return None
    gaps = [row_lines[index + 1] - row_lines[index] for index in range(len(row_lines) - 1)]
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    if median_gap <= 0:
        return None
    return row_lines[-1] + median_gap


def _best_band(box: BoundingBox, bands: Sequence[tuple[float, float]]) -> int:
    """Chỉ số cột của ``box``: ưu tiên cột chồng lấn nhiều nhất theo trục x.

    Band cột được suy từ token hàng header nên chỉ rộng bằng chữ trong header;
    token ở hàng dữ liệu hoàn toàn có thể **tràn ra ngoài mọi band** (VD giá trị
    dài hơn tiêu đề cột). Những token đó không chồng lấn band nào, khi ấy phải
    lấy cột **gần nhất theo khoảng cách** - nếu không chúng sẽ rơi về cột 0 và bị
    dính vào ô nhãn (bug thực tế: ``"Ưu đãi Data"`` nuốt luôn ``"tối đa 8gb/1
    ngày"``, khiến ``dataGB`` đọc ra 8 thay vì 60).
    """
    best_index = 0
    best_overlap = 0.0
    found_overlap = False
    for index, (start, end) in enumerate(bands):
        overlap = min(box.x2, end) - max(box.x1, start)
        if overlap > 0 and (not found_overlap or overlap > best_overlap):
            found_overlap = True
            best_overlap = overlap
            best_index = index
    if found_overlap:
        return best_index

    center_x = box.center[0]
    nearest_index = 0
    nearest_distance = float("inf")
    for index, (start, end) in enumerate(bands):
        if center_x < start:
            distance = start - center_x
        elif center_x > end:
            distance = center_x - end
        else:  # pragma: no cover - đã được nhánh chồng lấn ở trên xử lý
            distance = 0.0
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index
    return nearest_index


__all__ = [
    "MorphologyTableRecognizer",
    "TableMorphologyUnavailableError",
    "detect_grid",
]
