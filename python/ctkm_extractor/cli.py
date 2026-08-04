"""CLI: ``python -m ctkm_extractor.cli --image <path> --out result.json [--debug]``.

Chương trình luôn ghi ra JSON đủ field (giá trị thiếu là ``null``) kể cả khi OCR
lỗi, và trả exit code khác 0 chỉ khi lỗi cấu hình (schema hỏng, không có engine).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .extraction.extractor import CTKMExtractor, ExtractionResult, SchemaError
from .pdf import PdfUnavailableError, looks_like_pdf
from .ocr import (
    AUTO_ENGINE,
    ENGINE_PRIORITY,
    ProviderUnavailableError,
    available_engines,
)

logger = logging.getLogger("ctkm_extractor")

EXIT_OK = 0
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    """Khai báo tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        prog="python -m ctkm_extractor.cli",
        description="Trích xuất thông tin CTKM từ ảnh bảng biểu ra JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        help="Đường dẫn ảnh chứa bảng CTKM (đuôi .pdf được xử lý như --pdf)",
    )
    source.add_argument(
        "--pdf",
        help="Đường dẫn PDF nhiều trang; tách từng trang, OCR rồi gộp kết quả",
    )
    source.add_argument(
        "--text-file",
        help="Bỏ qua OCR, đọc thẳng raw text từ file (dùng để debug tầng trích xuất)",
    )
    parser.add_argument("--out", help="File JSON đầu ra; bỏ trống thì in ra stdout")
    parser.add_argument(
        "--engine",
        default="paddle_vietocr",
        choices=[*ENGINE_PRIORITY, AUTO_ENGINE],
        help="OCR engine: paddle_vietocr (PP-OCRv5 detect + VietOCR) hoặc tesseract",
    )
    parser.add_argument(
        "--schema",
        help="Đường dẫn schema.yaml tuỳ biến; mặc định dùng schema đi kèm package",
    )
    parser.add_argument(
        "--no-morphology",
        action="store_true",
        help="Bỏ qua dò bảng bằng đường kẻ (CV cổ điển), dùng thẳng PP-Structure",
    )
    parser.add_argument(
        "--no-pp-structure",
        action="store_true",
        help="Bỏ qua PP-Structure, dùng thẳng fallback cluster bounding box",
    )
    parser.add_argument(
        "--strict-engine",
        action="store_true",
        help="Không tự fallback sang engine khác khi engine yêu cầu thiếu dependency",
    )
    binarize = parser.add_mutually_exclusive_group()
    binarize.add_argument(
        "--no-binarize",
        action="store_true",
        help=(
            "Ép TẮT adaptive threshold. Mặc định chương trình chạy CẢ HAI cấu hình "
            "rồi giữ kết quả trích được nhiều field hơn; ép để chỉ chạy một lượt"
        ),
    )
    binarize.add_argument(
        "--binarize",
        action="store_true",
        help="Ép BẬT adaptive threshold, chỉ chạy một lượt",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Độ phân giải render mỗi trang PDF"
    )
    parser.add_argument(
        "--pages",
        help='Chọn trang PDF cần xử lý, VD "1,3-5"; bỏ trống = mọi trang',
    )
    parser.add_argument(
        "--keep-pages",
        action="store_true",
        help="Giữ lại ảnh từng trang đã render (soi khi debug)",
    )
    parser.add_argument("--indent", type=int, default=2, help="Số space thụt lề JSON")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="In OCR raw text, bảng đã dựng và nguồn của từng field ra stderr",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="In danh sách engine khả dụng trong môi trường hiện tại rồi thoát",
    )
    return parser


def configure_logging(debug: bool) -> None:
    """Bật log; ở chế độ debug thì hiện cả log DEBUG."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def write_output(result: ExtractionResult, out_path: str | None, indent: int) -> None:
    """Ghi JSON ra file hoặc stdout."""
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)
    if out_path:
        target = Path(out_path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
        logger.info("Đã ghi kết quả vào %s", target)
    else:
        print(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào CLI; trả về exit code."""
    parser = build_parser()
    # --list-engines không cần --image/--text-file nên xử lý trước khi parse chặt chẽ.
    if argv is None:
        argv = sys.argv[1:]
    if "--list-engines" in argv:
        configure_logging(False)
        engines = available_engines()
        print("Engine khả dụng:", ", ".join(engines) if engines else "(không có)")
        return EXIT_OK

    args = parser.parse_args(argv)
    configure_logging(args.debug)

    try:
        # None = tự chọn: chạy cả hai cấu hình rồi giữ kết quả nhiều field hơn.
        binarize = None
        if args.no_binarize:
            binarize = False
        elif args.binarize:
            binarize = True
        extractor = CTKMExtractor(
            args.schema,
            engine=args.engine,
            use_morphology=not args.no_morphology,
            use_pp_structure=not args.no_pp_structure,
            strict_engine=args.strict_engine,
            binarize=binarize,
        )
    except SchemaError as exc:
        logger.error("Schema lỗi: %s", exc)
        return EXIT_ERROR

    try:
        if args.text_file:
            text = Path(args.text_file).read_text(encoding="utf-8")
            result = extractor.extract_from_text(text)
        else:
            # --image trỏ vào .pdf cũng được xử lý như --pdf cho tiện.
            source_path = Path(args.pdf or args.image)
            if not source_path.is_file():
                logger.error("Không tìm thấy file đầu vào: %s", source_path)
                return EXIT_ERROR
            if args.pdf or looks_like_pdf(str(source_path)):
                result = extractor.extract_from_pdf(
                    str(source_path),
                    dpi=args.dpi,
                    pages=args.pages,
                    keep_pages=args.keep_pages,
                )
            else:
                result = extractor.extract_from_image(str(source_path))
    except ProviderUnavailableError as exc:
        logger.error(
            "Không khởi tạo được OCR engine (%s). Cài paddleocr+vietocr hoặc "
            "tesseract-ocr, hoặc chạy lại với --engine tesseract.",
            exc,
        )
        return EXIT_ERROR
    except PdfUnavailableError as exc:
        logger.error("Không xử lý được PDF: %s", exc)
        return EXIT_ERROR
    except OSError as exc:
        logger.error("Lỗi đọc dữ liệu đầu vào: %s", exc)
        return EXIT_ERROR

    if args.debug:
        print(result.debug_report(), file=sys.stderr)

    write_output(result, args.out, args.indent)

    missing = [name for name, item in result.fields.items() if not item.found]
    if missing:
        logger.warning("Các field không trích xuất được (null): %s", ", ".join(missing))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
