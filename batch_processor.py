"""
Legal Document OCR - Batch Processing Module
=============================================
Handles multiple files and folder processing
Supports recursive folder scanning and parallel processing
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from pdf_processor import PDFProcessor, PDFInfo
from ocr_engine import HybridOCRPipeline, OCRResult
from gemini_corrector import GeminiCorrector, CorrectionResult
from config import get_settings


@dataclass
class FileInfo:
    """Information about a file to process"""
    path: str
    filename: str
    file_type: str  # 'pdf', 'image'
    size_mb: float
    page_count: int = 1
    status: str = 'pending'  # pending, processing, completed, failed
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Result of batch processing"""
    total_files: int
    processed_files: int
    failed_files: int
    total_pages: int
    total_time: float
    files: List[Dict] = field(default_factory=list)
    output_dir: str = ""


@dataclass
class FolderScanResult:
    """Result of folder scanning"""
    folder_path: str
    pdf_files: List[str]
    image_files: List[str]
    total_files: int
    total_size_mb: float
    estimated_pages: int


class BatchProcessor:
    """
    Batch processing system for multiple files and folders.

    Features:
    - Multiple file processing
    - Recursive folder scanning
    - Parallel processing with thread pool
    - Progress callback support
    - Error recovery and logging
    - Output organization
    """

    # Supported file extensions
    PDF_EXTENSIONS = {'.pdf'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}

    def __init__(
        self,
        got_model_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        output_base_dir: Optional[str] = None,
        max_workers: int = 1,
        use_gemini: bool = True
    ):
        """
        Initialize batch processor.

        Args:
            got_model_path: Path to GOT-OCR model
            yolo_model_path: Path to YOLO model
            output_base_dir: Base directory for outputs
            max_workers: Maximum parallel workers
            use_gemini: Whether to use Gemini AI correction
        """
        settings = get_settings()

        self.got_model_path = got_model_path or settings.got_model_path
        self.yolo_model_path = yolo_model_path or settings.yolo_model_path
        self.output_base_dir = output_base_dir or settings.output_dir
        self.max_workers = max_workers
        self.use_gemini = use_gemini

        # Initialize components lazily
        self._ocr_pipeline = None
        self._pdf_processor = None
        self._gemini_corrector = None

        # Ensure output directory exists
        os.makedirs(self.output_base_dir, exist_ok=True)

        logger.info(f"BatchProcessor initialized: output_dir={self.output_base_dir}")

    @property
    def ocr_pipeline(self) -> HybridOCRPipeline:
        """Lazy load OCR pipeline"""
        if self._ocr_pipeline is None:
            self._ocr_pipeline = HybridOCRPipeline(
                got_model_path=self.got_model_path,
                yolo_model_path=self.yolo_model_path
            )
        return self._ocr_pipeline

    @property
    def pdf_processor(self) -> PDFProcessor:
        """Lazy load PDF processor"""
        if self._pdf_processor is None:
            self._pdf_processor = PDFProcessor()
        return self._pdf_processor

    @property
    def gemini_corrector(self) -> GeminiCorrector:
        """Lazy load Gemini corrector"""
        if self._gemini_corrector is None:
            self._gemini_corrector = GeminiCorrector()
        return self._gemini_corrector

    def scan_folder(
        self,
        folder_path: str,
        recursive: bool = True
    ) -> FolderScanResult:
        """
        Scan a folder for processable files.

        Args:
            folder_path: Path to folder
            recursive: Whether to scan subfolders

        Returns:
            FolderScanResult with found files
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        pdf_files = []
        image_files = []
        total_size = 0

        # Scan patterns
        pattern = "**/*" if recursive else "*"

        for file_path in folder.glob(pattern):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            size = file_path.stat().st_size

            if ext in self.PDF_EXTENSIONS:
                pdf_files.append(str(file_path))
                total_size += size
            elif ext in self.IMAGE_EXTENSIONS:
                image_files.append(str(file_path))
                total_size += size

        # Estimate pages (1 page per image, need to check PDFs)
        estimated_pages = len(image_files)

        for pdf_path in pdf_files:
            try:
                info = self.pdf_processor.get_pdf_info(pdf_path)
                estimated_pages += info.page_count
            except Exception:
                estimated_pages += 10  # Default estimate

        return FolderScanResult(
            folder_path=str(folder),
            pdf_files=sorted(pdf_files),
            image_files=sorted(image_files),
            total_files=len(pdf_files) + len(image_files),
            total_size_mb=total_size / (1024 * 1024),
            estimated_pages=estimated_pages
        )

    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a single file.

        Args:
            file_path: Path to file

        Returns:
            FileInfo object
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        size_mb = path.stat().st_size / (1024 * 1024)

        if ext in self.PDF_EXTENSIONS:
            try:
                info = self.pdf_processor.get_pdf_info(file_path)
                page_count = info.page_count
            except Exception:
                page_count = 1
            file_type = 'pdf'
        elif ext in self.IMAGE_EXTENSIONS:
            page_count = 1
            file_type = 'image'
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return FileInfo(
            path=str(path),
            filename=path.name,
            file_type=file_type,
            size_mb=size_mb,
            page_count=page_count
        )

    def _create_output_dir(self, input_path: str, batch_id: str) -> str:
        """Create output directory for a file"""
        filename = Path(input_path).stem
        output_dir = os.path.join(
            self.output_base_dir,
            batch_id,
            filename
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_single_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a single file (PDF or image).

        Args:
            file_path: Path to file
            output_dir: Output directory (auto-generated if None)
            progress_callback: Callback(current_page, total_pages, status)

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        file_info = self.get_file_info(file_path)

        if output_dir is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self._create_output_dir(file_path, batch_id)

        result = {
            "input_file": file_path,
            "output_dir": output_dir,
            "file_type": file_info.file_type,
            "page_count": file_info.page_count,
            "status": "processing",
            "pages": [],
            "markdown_file": None,
            "json_file": None,
            "error": None
        }

        try:
            all_markdown = []
            all_results = []

            if file_info.file_type == 'pdf':
                # Process PDF pages
                for page_image in self.pdf_processor.iter_pages(file_path):
                    if progress_callback:
                        progress_callback(
                            page_image.page_number,
                            file_info.page_count,
                            f"Processing page {page_image.page_number}"
                        )

                    # OCR processing
                    ocr_result = self.ocr_pipeline.process_image(
                        page_image.image,
                        page_image.page_number
                    )

                    # Gemini correction
                    if self.use_gemini:
                        correction = self.gemini_corrector.correct(ocr_result.markdown)
                        final_markdown = correction.corrected_text
                    else:
                        final_markdown = ocr_result.markdown

                    all_markdown.append(f"## 페이지 {page_image.page_number}\n\n{final_markdown}")
                    all_results.append({
                        "page": page_image.page_number,
                        "markdown": final_markdown,
                        "tables_count": ocr_result.tables_count,
                        "confidence": ocr_result.confidence
                    })

            else:
                # Process single image
                if progress_callback:
                    progress_callback(1, 1, "Processing image")

                ocr_result = self.ocr_pipeline.process_image(file_path, 1)

                if self.use_gemini:
                    correction = self.gemini_corrector.correct(ocr_result.markdown)
                    final_markdown = correction.corrected_text
                else:
                    final_markdown = ocr_result.markdown

                all_markdown.append(final_markdown)
                all_results.append({
                    "page": 1,
                    "markdown": final_markdown,
                    "tables_count": ocr_result.tables_count,
                    "confidence": ocr_result.confidence
                })

            # Save results
            combined_markdown = "\n\n---\n\n".join(all_markdown)

            # Save markdown file
            md_filename = Path(file_path).stem + ".md"
            md_path = os.path.join(output_dir, md_filename)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(combined_markdown)
            result["markdown_file"] = md_path

            # Save JSON file
            json_filename = Path(file_path).stem + ".json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_file": file_path,
                    "processed_at": datetime.now().isoformat(),
                    "page_count": len(all_results),
                    "pages": all_results
                }, f, ensure_ascii=False, indent=2)
            result["json_file"] = json_path

            result["pages"] = all_results
            result["status"] = "completed"
            result["processing_time"] = time.time() - start_time

            logger.info(f"Processed {file_path}: {len(all_results)} pages in {result['processing_time']:.2f}s")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time

        return result

    def process_multiple_files(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> BatchResult:
        """
        Process multiple files.

        Args:
            file_paths: List of file paths
            progress_callback: Callback(file_index, total_files, filename, status)

        Returns:
            BatchResult with all processing results
        """
        start_time = time.time()
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = []
        processed = 0
        failed = 0
        total_pages = 0

        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i + 1, len(file_paths), Path(file_path).name, "processing")

            try:
                output_dir = self._create_output_dir(file_path, batch_id)
                result = self.process_single_file(file_path, output_dir)
                results.append(result)

                if result["status"] == "completed":
                    processed += 1
                    total_pages += result["page_count"]
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    "input_file": file_path,
                    "status": "failed",
                    "error": str(e)
                })
                failed += 1

            if progress_callback:
                status = "completed" if results[-1].get("status") == "completed" else "failed"
                progress_callback(i + 1, len(file_paths), Path(file_path).name, status)

        return BatchResult(
            total_files=len(file_paths),
            processed_files=processed,
            failed_files=failed,
            total_pages=total_pages,
            total_time=time.time() - start_time,
            files=results,
            output_dir=os.path.join(self.output_base_dir, batch_id)
        )

    def process_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> BatchResult:
        """
        Process all files in a folder.

        Args:
            folder_path: Path to folder
            recursive: Whether to process subfolders
            progress_callback: Callback(file_index, total_files, filename, status)

        Returns:
            BatchResult with all processing results
        """
        # Scan folder
        scan_result = self.scan_folder(folder_path, recursive)

        logger.info(
            f"Found {scan_result.total_files} files in {folder_path}: "
            f"{len(scan_result.pdf_files)} PDFs, {len(scan_result.image_files)} images"
        )

        # Combine all files
        all_files = scan_result.pdf_files + scan_result.image_files

        if not all_files:
            return BatchResult(
                total_files=0,
                processed_files=0,
                failed_files=0,
                total_pages=0,
                total_time=0,
                files=[],
                output_dir=self.output_base_dir
            )

        # Process all files
        return self.process_multiple_files(all_files, progress_callback)

    def cleanup(self) -> None:
        """Clean up resources"""
        if self._ocr_pipeline:
            self._ocr_pipeline.cleanup()
        if self._pdf_processor:
            self._pdf_processor.cleanup()

        logger.info("BatchProcessor cleanup complete")


# Convenience functions
def process_files(
    file_paths: List[str],
    output_dir: Optional[str] = None,
    use_gemini: bool = True
) -> BatchResult:
    """
    Convenience function to process multiple files.

    Args:
        file_paths: List of file paths
        output_dir: Output directory
        use_gemini: Whether to use Gemini correction

    Returns:
        BatchResult
    """
    processor = BatchProcessor(
        output_base_dir=output_dir,
        use_gemini=use_gemini
    )
    try:
        return processor.process_multiple_files(file_paths)
    finally:
        processor.cleanup()


def process_folder(
    folder_path: str,
    output_dir: Optional[str] = None,
    recursive: bool = True,
    use_gemini: bool = True
) -> BatchResult:
    """
    Convenience function to process a folder.

    Args:
        folder_path: Path to folder
        output_dir: Output directory
        recursive: Whether to process subfolders
        use_gemini: Whether to use Gemini correction

    Returns:
        BatchResult
    """
    processor = BatchProcessor(
        output_base_dir=output_dir,
        use_gemini=use_gemini
    )
    try:
        return processor.process_folder(folder_path, recursive)
    finally:
        processor.cleanup()
