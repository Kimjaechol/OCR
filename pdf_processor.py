"""
Legal Document OCR - PDF Processing Module
==========================================
Handles PDF to image conversion with batching for large documents
Supports up to 5000+ pages with memory-efficient processing
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Generator, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    logger.warning("pdf2image not installed. PDF processing will be unavailable.")
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF not installed. Alternative PDF processing unavailable.")
    PYMUPDF_AVAILABLE = False

import cv2
import numpy as np


@dataclass
class PDFInfo:
    """Information about a PDF document"""
    path: str
    page_count: int
    file_size_mb: float
    estimated_processing_time: float  # in seconds


@dataclass
class PageImage:
    """Represents a single page image"""
    page_number: int
    image: np.ndarray
    width: int
    height: int


class PDFProcessor:
    """
    Memory-efficient PDF processor for large documents.

    Features:
    - Batch processing to handle 5000+ pages
    - Memory management with image cleanup
    - Progress tracking support
    - Multiple PDF library fallback (pdf2image -> PyMuPDF)
    """

    def __init__(
        self,
        dpi: int = 300,
        batch_size: int = 50,
        temp_dir: Optional[str] = None,
        use_poppler: bool = True
    ):
        """
        Initialize PDF processor.

        Args:
            dpi: Resolution for PDF rendering (300 recommended for OCR)
            batch_size: Number of pages to process at once
            temp_dir: Directory for temporary files
            use_poppler: Use poppler (pdf2image) if available
        """
        self.dpi = dpi
        self.batch_size = batch_size
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="ocr_pdf_")
        self.use_poppler = use_poppler

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        logger.info(
            f"PDFProcessor initialized: DPI={dpi}, batch_size={batch_size}"
        )

    def get_pdf_info(self, pdf_path: str) -> PDFInfo:
        """
        Get information about a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFInfo object with document details
        """
        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_size_mb = path.stat().st_size / (1024 * 1024)

        # Get page count
        page_count = self._get_page_count(pdf_path)

        # Estimate processing time (rough estimate: 2 seconds per page)
        estimated_time = page_count * 2.0

        return PDFInfo(
            path=str(path),
            page_count=page_count,
            file_size_mb=file_size_mb,
            estimated_processing_time=estimated_time
        )

    def _get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF"""
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(pdf_path)
                count = len(doc)
                doc.close()
                return count
            except Exception as e:
                logger.warning(f"PyMuPDF page count failed: {e}")

        if PDF2IMAGE_AVAILABLE:
            try:
                from pdf2image.pdf2image import pdfinfo_from_path
                info = pdfinfo_from_path(pdf_path)
                return info.get('Pages', 0)
            except Exception as e:
                logger.warning(f"pdf2image page count failed: {e}")

        raise RuntimeError("No PDF library available to count pages")

    def convert_page_to_image(
        self,
        pdf_path: str,
        page_number: int
    ) -> PageImage:
        """
        Convert a single PDF page to image.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            PageImage object
        """
        if self.use_poppler and PDF2IMAGE_AVAILABLE:
            return self._convert_with_poppler(pdf_path, page_number)
        elif PYMUPDF_AVAILABLE:
            return self._convert_with_pymupdf(pdf_path, page_number)
        else:
            raise RuntimeError("No PDF library available")

    def _convert_with_poppler(
        self,
        pdf_path: str,
        page_number: int
    ) -> PageImage:
        """Convert using pdf2image (poppler)"""
        images = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            first_page=page_number,
            last_page=page_number,
            fmt='RGB'
        )

        if not images:
            raise ValueError(f"Failed to convert page {page_number}")

        pil_image = images[0]

        # Convert PIL to OpenCV format (BGR)
        np_image = np.array(pil_image)
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        return PageImage(
            page_number=page_number,
            image=cv_image,
            width=cv_image.shape[1],
            height=cv_image.shape[0]
        )

    def _convert_with_pymupdf(
        self,
        pdf_path: str,
        page_number: int
    ) -> PageImage:
        """Convert using PyMuPDF"""
        doc = fitz.open(pdf_path)

        try:
            # PyMuPDF uses 0-indexed pages
            page = doc[page_number - 1]

            # Calculate zoom for desired DPI (default is 72 DPI)
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)

            # Convert to numpy array
            np_image = np.frombuffer(pix.samples, dtype=np.uint8)
            np_image = np_image.reshape(pix.height, pix.width, pix.n)

            # Convert to BGR if needed
            if pix.n == 4:  # RGBA
                cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            else:  # Grayscale
                cv_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

            return PageImage(
                page_number=page_number,
                image=cv_image,
                width=cv_image.shape[1],
                height=cv_image.shape[0]
            )
        finally:
            doc.close()

    def convert_batch(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int
    ) -> List[PageImage]:
        """
        Convert a batch of PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            start_page: First page (1-indexed)
            end_page: Last page (inclusive)

        Returns:
            List of PageImage objects
        """
        pages = []

        if self.use_poppler and PDF2IMAGE_AVAILABLE:
            # Batch convert with poppler for efficiency
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    first_page=start_page,
                    last_page=end_page,
                    fmt='RGB'
                )

                for i, pil_image in enumerate(images):
                    np_image = np.array(pil_image)
                    cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

                    pages.append(PageImage(
                        page_number=start_page + i,
                        image=cv_image,
                        width=cv_image.shape[1],
                        height=cv_image.shape[0]
                    ))

                return pages

            except Exception as e:
                logger.warning(f"Batch convert failed: {e}, falling back to single")

        # Fallback to single page conversion
        for page_num in range(start_page, end_page + 1):
            try:
                page_image = self.convert_page_to_image(pdf_path, page_num)
                pages.append(page_image)
            except Exception as e:
                logger.error(f"Failed to convert page {page_num}: {e}")

        return pages

    def iter_pages(
        self,
        pdf_path: str,
        start_page: int = 1,
        end_page: Optional[int] = None
    ) -> Generator[PageImage, None, None]:
        """
        Iterate through PDF pages with memory-efficient batching.

        Args:
            pdf_path: Path to PDF file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (None = all pages)

        Yields:
            PageImage objects one at a time
        """
        info = self.get_pdf_info(pdf_path)
        total_pages = info.page_count

        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(end_page, total_pages)

        logger.info(f"Processing pages {start_page} to {end_page} of {total_pages}")

        current_page = start_page

        while current_page <= end_page:
            # Calculate batch end
            batch_end = min(current_page + self.batch_size - 1, end_page)

            # Convert batch
            batch_pages = self.convert_batch(pdf_path, current_page, batch_end)

            # Yield pages
            for page in batch_pages:
                yield page

            current_page = batch_end + 1

            # Memory cleanup hint
            import gc
            gc.collect()

    def save_page_image(
        self,
        page_image: PageImage,
        output_dir: str
    ) -> str:
        """
        Save a page image to disk.

        Args:
            page_image: PageImage to save
            output_dir: Output directory

        Returns:
            Path to saved image
        """
        os.makedirs(output_dir, exist_ok=True)

        filename = f"page_{page_image.page_number:05d}.png"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, page_image.image)

        return filepath

    def extract_all_pages(
        self,
        pdf_path: str,
        output_dir: str,
        start_page: int = 1,
        end_page: Optional[int] = None
    ) -> List[str]:
        """
        Extract all pages from PDF and save as images.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            start_page: First page to extract
            end_page: Last page to extract

        Returns:
            List of saved image paths
        """
        saved_paths = []

        for page in self.iter_pages(pdf_path, start_page, end_page):
            path = self.save_page_image(page, output_dir)
            saved_paths.append(path)
            logger.debug(f"Saved page {page.page_number}")

        logger.info(f"Extracted {len(saved_paths)} pages to {output_dir}")

        return saved_paths

    def cleanup(self) -> None:
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Convenience function to get PDF page count.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages
    """
    processor = PDFProcessor()
    info = processor.get_pdf_info(pdf_path)
    return info.page_count


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300
) -> List[str]:
    """
    Convenience function to convert PDF to images.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for images
        dpi: Resolution for conversion

    Returns:
        List of image file paths
    """
    processor = PDFProcessor(dpi=dpi)
    return processor.extract_all_pages(pdf_path, output_dir)
