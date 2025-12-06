"""
Legal Document OCR - Command Line Interface
============================================
CLI for testing OCR pipeline without running the full server
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


def process_single_file(
    file_path: str,
    output_dir: Optional[str] = None,
    use_gemini: bool = True,
    verbose: bool = False
) -> dict:
    """Process a single file (PDF or image)"""
    from batch_processor import BatchProcessor

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    processor = BatchProcessor(
        output_base_dir=output_dir or "./output",
        use_gemini=use_gemini
    )

    try:
        def progress_callback(current, total, status):
            logger.info(f"진행: {current}/{total} - {status}")

        result = processor.process_single_file(
            file_path,
            progress_callback=progress_callback
        )

        return result
    finally:
        processor.cleanup()


def process_folder(
    folder_path: str,
    output_dir: Optional[str] = None,
    recursive: bool = True,
    use_gemini: bool = True,
    verbose: bool = False
) -> dict:
    """Process all files in a folder"""
    from batch_processor import BatchProcessor

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    processor = BatchProcessor(
        output_base_dir=output_dir or "./output",
        use_gemini=use_gemini
    )

    try:
        # First scan
        scan_result = processor.scan_folder(folder_path, recursive)
        logger.info(f"폴더 스캔 완료: {scan_result.total_files}개 파일, ~{scan_result.estimated_pages}페이지")

        if scan_result.total_files == 0:
            logger.warning("처리할 파일이 없습니다.")
            return {"status": "no_files", "total_files": 0}

        def progress_callback(current, total, filename, status):
            logger.info(f"[{current}/{total}] {filename} - {status}")

        result = processor.process_folder(
            folder_path,
            recursive=recursive,
            progress_callback=progress_callback
        )

        return {
            "status": "completed",
            "total_files": result.total_files,
            "processed_files": result.processed_files,
            "failed_files": result.failed_files,
            "total_pages": result.total_pages,
            "total_time": result.total_time,
            "output_dir": result.output_dir
        }
    finally:
        processor.cleanup()


def scan_folder(folder_path: str, recursive: bool = True) -> dict:
    """Scan folder and show file statistics"""
    from batch_processor import BatchProcessor
    from config import get_settings

    settings = get_settings()
    processor = BatchProcessor()

    scan_result = processor.scan_folder(folder_path, recursive)

    return {
        "folder_path": scan_result.folder_path,
        "pdf_files": len(scan_result.pdf_files),
        "image_files": len(scan_result.image_files),
        "total_files": scan_result.total_files,
        "total_size_mb": round(scan_result.total_size_mb, 2),
        "estimated_pages": scan_result.estimated_pages,
        "estimated_price_krw": scan_result.estimated_pages * settings.price_per_page
    }


def test_ocr_engine(image_path: str) -> dict:
    """Test OCR engine on a single image"""
    from ocr_engine import HybridOCRPipeline
    from config import get_settings

    settings = get_settings()

    logger.info("OCR 엔진 초기화 중...")
    pipeline = HybridOCRPipeline(
        got_model_path=settings.got_model_path,
        yolo_model_path=settings.yolo_model_path
    )

    try:
        logger.info(f"이미지 처리 중: {image_path}")
        result = pipeline.process_image(image_path, page_number=1)

        return {
            "markdown": result.markdown,
            "raw_text": result.raw_text,
            "tables_count": result.tables_count,
            "confidence": result.confidence,
            "processing_time": result.processing_time
        }
    finally:
        pipeline.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="법률 문서 OCR CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 파일 처리
  python cli.py process contract.pdf -o ./output

  # 폴더 전체 처리
  python cli.py folder ./documents -o ./output --recursive

  # 폴더 스캔 (처리 전 확인)
  python cli.py scan ./documents

  # OCR 엔진 테스트
  python cli.py test image.png
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # process command
    process_parser = subparsers.add_parser("process", help="단일 파일 처리")
    process_parser.add_argument("file", help="처리할 파일 경로 (PDF 또는 이미지)")
    process_parser.add_argument("-o", "--output", help="출력 디렉토리", default="./output")
    process_parser.add_argument("--no-gemini", action="store_true", help="Gemini 교정 비활성화")
    process_parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그 출력")

    # folder command
    folder_parser = subparsers.add_parser("folder", help="폴더 전체 처리")
    folder_parser.add_argument("path", help="처리할 폴더 경로")
    folder_parser.add_argument("-o", "--output", help="출력 디렉토리", default="./output")
    folder_parser.add_argument("--no-recursive", action="store_true", help="하위 폴더 제외")
    folder_parser.add_argument("--no-gemini", action="store_true", help="Gemini 교정 비활성화")
    folder_parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그 출력")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="폴더 스캔 (처리 전 확인)")
    scan_parser.add_argument("path", help="스캔할 폴더 경로")
    scan_parser.add_argument("--no-recursive", action="store_true", help="하위 폴더 제외")

    # test command
    test_parser = subparsers.add_parser("test", help="OCR 엔진 테스트")
    test_parser.add_argument("image", help="테스트할 이미지 파일")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "process":
            if not os.path.exists(args.file):
                logger.error(f"파일을 찾을 수 없습니다: {args.file}")
                sys.exit(1)

            logger.info(f"파일 처리 시작: {args.file}")
            start_time = time.time()

            result = process_single_file(
                args.file,
                output_dir=args.output,
                use_gemini=not args.no_gemini,
                verbose=args.verbose
            )

            elapsed = time.time() - start_time

            if result.get("status") == "completed":
                logger.info(f"처리 완료! ({elapsed:.2f}초)")
                logger.info(f"출력 디렉토리: {result.get('output_dir')}")
                logger.info(f"마크다운 파일: {result.get('markdown_file')}")
                logger.info(f"JSON 파일: {result.get('json_file')}")
            else:
                logger.error(f"처리 실패: {result.get('error')}")
                sys.exit(1)

        elif args.command == "folder":
            if not os.path.exists(args.path):
                logger.error(f"폴더를 찾을 수 없습니다: {args.path}")
                sys.exit(1)

            logger.info(f"폴더 처리 시작: {args.path}")
            start_time = time.time()

            result = process_folder(
                args.path,
                output_dir=args.output,
                recursive=not args.no_recursive,
                use_gemini=not args.no_gemini,
                verbose=args.verbose
            )

            elapsed = time.time() - start_time

            if result.get("status") == "completed":
                logger.info(f"처리 완료! ({elapsed:.2f}초)")
                logger.info(f"총 파일: {result.get('total_files')}개")
                logger.info(f"성공: {result.get('processed_files')}개")
                logger.info(f"실패: {result.get('failed_files')}개")
                logger.info(f"총 페이지: {result.get('total_pages')}페이지")
                logger.info(f"출력 디렉토리: {result.get('output_dir')}")
            elif result.get("status") == "no_files":
                logger.warning("처리할 파일이 없습니다.")
            else:
                logger.error(f"처리 실패")
                sys.exit(1)

        elif args.command == "scan":
            if not os.path.exists(args.path):
                logger.error(f"폴더를 찾을 수 없습니다: {args.path}")
                sys.exit(1)

            result = scan_folder(args.path, recursive=not args.no_recursive)

            print("\n" + "=" * 50)
            print(f"폴더: {result['folder_path']}")
            print("=" * 50)
            print(f"PDF 파일: {result['pdf_files']}개")
            print(f"이미지 파일: {result['image_files']}개")
            print(f"총 파일: {result['total_files']}개")
            print(f"총 용량: {result['total_size_mb']} MB")
            print(f"예상 페이지: ~{result['estimated_pages']}페이지")
            print(f"예상 비용: {result['estimated_price_krw']:,}원")
            print("=" * 50 + "\n")

        elif args.command == "test":
            if not os.path.exists(args.image):
                logger.error(f"파일을 찾을 수 없습니다: {args.image}")
                sys.exit(1)

            result = test_ocr_engine(args.image)

            print("\n" + "=" * 50)
            print("OCR 결과")
            print("=" * 50)
            print(f"표 개수: {result['tables_count']}")
            print(f"신뢰도: {result['confidence']:.2f}")
            print(f"처리 시간: {result['processing_time']:.2f}초")
            print("-" * 50)
            print("마크다운 출력:")
            print("-" * 50)
            print(result['markdown'][:2000])
            if len(result['markdown']) > 2000:
                print(f"\n... (총 {len(result['markdown'])} 문자)")
            print("=" * 50 + "\n")

    except KeyboardInterrupt:
        logger.warning("사용자가 중단했습니다.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
