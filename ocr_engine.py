"""
Legal Document OCR - OCR Engine Module
=======================================
Hybrid OCR engine using GOT-OCR (tables) + PaddleOCR (text)
with legal document specialized parsing and correction
"""

import cv2
import numpy as np
import re
import statistics
import os
import tempfile
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

import torch

try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.warning("PaddleOCR not installed")
    PaddleOCR = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.warning("transformers not installed")
    AutoModelForCausalLM = None
    AutoTokenizer = None

from layout_analyzer import LayoutAnalyzer, DetectedRegion, TextFormatInfo
from html_generator import HTMLGenerator, StyledText, TextStyle, DocumentFormatter


@dataclass
class TextSegment:
    """Represents a parsed text segment with formatting info"""
    category: str  # 'text' or 'table'
    y_pos: int
    text: str
    tag: str = "p"  # 'h1', 'h2', 'p'
    is_bold: bool = False
    is_indented: bool = False


@dataclass
class OCRResult:
    """Complete OCR result for a document page"""
    page_number: int
    raw_text: str
    markdown: str
    html: str = ""  # HTML formatted output
    segments: List[TextSegment] = field(default_factory=list)
    tables_count: int = 0
    confidence: float = 0.0
    processing_time: float = 0.0


class LegalTextParser:
    """
    Legal document text parser with structure analysis.

    Features:
    - Heading detection (h1, h2 based on font size)
    - Bold text detection using pixel density
    - Indentation detection
    - Legal-specific typo correction
    """

    # Legal document style thresholds
    H1_SCALE = 1.4       # 1.4x median height = h1
    H2_SCALE = 1.15      # 1.15x median height = h2
    INDENT_PX = 20       # 20px from baseline = indented
    BOLD_RATIO = 1.10    # 10% denser = bold

    # Legal typo patterns
    LEGAL_TYPO_PATTERNS = [
        # 갑(甲) 뒤의 오타 복원
        (r'\bZ\b', '乙'),
        (r'\bE\b', '乙'),
        # 법조문 숫자 오타
        (r'(제\s*\d+)[oO](조)', r'\g<1>0\g<2>'),
        # 띄어쓰기 오류
        (r'제(\d+)조', r'제\g<1>조'),
    ]

    # 로마 숫자 매핑
    ROMAN_NUMERALS = {
        'I': 'Ⅰ', 'II': 'Ⅱ', 'III': 'Ⅲ',
        'IV': 'Ⅳ', 'V': 'Ⅴ', 'VI': 'Ⅵ',
        'VII': 'Ⅶ', 'VIII': 'Ⅷ', 'IX': 'Ⅸ', 'X': 'Ⅹ'
    }

    def __init__(self, image: np.ndarray):
        """
        Initialize parser with grayscale image.

        Args:
            image: Grayscale OpenCV image for density analysis
        """
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image

    def get_geometry(self, box: List) -> Tuple[int, int, int, int, np.ndarray]:
        """
        Extract geometric information from PaddleOCR box.

        Args:
            box: PaddleOCR box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Tuple of (height, x, y, width, points_array)
        """
        pts = np.array(box, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        return h, x, y, w, pts

    def calculate_density(self, pts: np.ndarray) -> float:
        """
        Calculate pixel density of text region.

        Args:
            pts: Polygon points of text region

        Returns:
            Pixel density (0-1)
        """
        try:
            mask = np.zeros_like(self.image)
            cv2.fillPoly(mask, [pts], 255)

            masked = cv2.bitwise_and(self.image, self.image, mask=mask)
            _, binary = cv2.threshold(
                masked, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            x, y, w, h = cv2.boundingRect(pts)
            if w * h == 0:
                return 0.0

            roi = binary[y:y+h, x:x+w]
            return cv2.countNonZero(roi) / (w * h)
        except Exception:
            return 0.0

    def is_bold(self, pts: np.ndarray, median_density: float) -> bool:
        """
        Determine if text is bold based on pixel density.

        Args:
            pts: Polygon points
            median_density: Median density of document

        Returns:
            True if text appears bold
        """
        if median_density == 0:
            return False

        density = self.calculate_density(pts)
        return density > (median_density * self.BOLD_RATIO)

    def fix_typos(self, text: str) -> str:
        """
        Fix legal document specific typos.

        Args:
            text: Raw OCR text

        Returns:
            Corrected text
        """
        # Apply regex patterns
        for pattern, replacement in self.LEGAL_TYPO_PATTERNS:
            if '갑' in text or '甲' in text:
                text = re.sub(pattern, replacement, text)

        # Roman numeral conversion
        words = text.split()
        corrected_words = []
        for word in words:
            if word in self.ROMAN_NUMERALS:
                corrected_words.append(self.ROMAN_NUMERALS[word])
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def parse_segments(self, ocr_result: List) -> List[TextSegment]:
        """
        Parse PaddleOCR results into structured segments.

        Args:
            ocr_result: PaddleOCR result list

        Returns:
            List of TextSegment objects
        """
        if not ocr_result:
            return []

        # Collect statistics
        heights = []
        densities = []
        x_positions = []

        for line in ocr_result:
            box = line[0]
            h, x, y, w, pts = self.get_geometry(box)
            heights.append(h)
            x_positions.append(x)

            density = self.calculate_density(pts)
            if density > 0:
                densities.append(density)

        # Calculate medians
        median_h = statistics.median(heights) if heights else 20
        min_x = min(x_positions) if x_positions else 0
        median_d = statistics.median(densities) if densities else 0.5

        # Parse each line
        segments = []
        for line in ocr_result:
            box = line[0]
            raw_text = line[1][0]
            confidence = line[1][1] if len(line[1]) > 1 else 0.0

            h, x, y, w, pts = self.get_geometry(box)

            # Apply typo correction
            text = self.fix_typos(raw_text)

            # Determine attributes
            ratio = h / median_h if median_h > 0 else 1
            is_indented = (x - min_x) >= self.INDENT_PX
            is_bold = self.is_bold(pts, median_d)

            # Determine tag
            tag = "p"
            if ratio >= self.H1_SCALE:
                tag = "h1"
            elif ratio >= self.H2_SCALE:
                tag = "h2"

            segments.append(TextSegment(
                category="text",
                y_pos=y,
                text=text,
                tag=tag,
                is_bold=is_bold,
                is_indented=is_indented
            ))

        return segments


class HybridOCRPipeline:
    """
    Hybrid OCR pipeline combining multiple OCR engines.

    Architecture:
    1. LayoutAnalyzer (YOLO) - Split tables and text
    2. PaddleOCR - Korean text recognition
    3. GOT-OCR - Table structure recognition
    4. LegalTextParser - Structure analysis

    Features:
    - GPU acceleration
    - Batch processing
    - Error recovery
    - Memory management
    """

    def __init__(
        self,
        got_model_path: str,
        yolo_model_path: str,
        use_gpu: bool = True,
        paddle_lang: str = 'korean'
    ):
        """
        Initialize the hybrid OCR pipeline.

        Args:
            got_model_path: Path to GOT-OCR model
            yolo_model_path: Path to YOLO model for table detection
            use_gpu: Whether to use GPU acceleration
            paddle_lang: Language for PaddleOCR
        """
        self.got_model_path = got_model_path
        self.yolo_model_path = yolo_model_path
        self.use_gpu = use_gpu
        self.paddle_lang = paddle_lang

        self.layout_analyzer = None
        self.paddle_ocr = None
        self.got_model = None
        self.got_tokenizer = None

        self._initialized = False

    def initialize(self) -> None:
        """Initialize all models (lazy loading)"""
        if self._initialized:
            return

        logger.info("Initializing OCR Pipeline...")

        # 1. Layout Analyzer
        logger.info("Loading Layout Analyzer (YOLO)...")
        self.layout_analyzer = LayoutAnalyzer(
            model_path=self.yolo_model_path,
            use_gpu=self.use_gpu
        )

        # 2. PaddleOCR
        if PaddleOCR is not None:
            logger.info("Loading PaddleOCR...")
            self.paddle_ocr = PaddleOCR(
                lang=self.paddle_lang,
                use_angle_cls=True,
                show_log=False,
                use_gpu=self.use_gpu
            )
        else:
            logger.warning("PaddleOCR not available")

        # 3. GOT-OCR
        if AutoModelForCausalLM is not None and AutoTokenizer is not None:
            logger.info("Loading GOT-OCR model...")
            try:
                self._load_got_model()
            except Exception as e:
                logger.error(f"Failed to load GOT-OCR: {e}")
                self.got_model = None
        else:
            logger.warning("GOT-OCR dependencies not available")

        self._initialized = True
        logger.info("OCR Pipeline initialized successfully")

    def _load_got_model(self) -> None:
        """Load GOT-OCR model with proper configuration"""
        model_path = Path(self.got_model_path)

        if not model_path.exists():
            logger.warning(f"GOT model path not found: {model_path}")
            return

        self.got_tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )

        device_map = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'

        self.got_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if device_map == 'cuda' else torch.float32
        )

        if device_map == 'cuda':
            self.got_model = self.got_model.eval().cuda()
        else:
            self.got_model = self.got_model.eval()

        logger.info(f"GOT-OCR loaded on {device_map}")

    def _process_table(self, table_image: np.ndarray) -> str:
        """
        Process table image with GOT-OCR.

        Args:
            table_image: BGR image of table region

        Returns:
            Markdown formatted table text
        """
        if self.got_model is None or self.got_tokenizer is None:
            return "[표 인식 불가: GOT-OCR 미로드]"

        # Save to temp file (GOT-OCR requires file path)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, table_image)

        try:
            result = self.got_model.chat(
                self.got_tokenizer,
                temp_path,
                ocr_type='format'
            )
            return result
        except Exception as e:
            logger.error(f"GOT-OCR error: {e}")
            return "[표 인식 중 오류 발생]"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _process_text(
        self,
        text_image: np.ndarray
    ) -> Tuple[List[TextSegment], float]:
        """
        Process text image with PaddleOCR.

        Args:
            text_image: BGR image with text (tables removed)

        Returns:
            Tuple of (parsed segments, average confidence)
        """
        if self.paddle_ocr is None:
            return [], 0.0

        try:
            results = self.paddle_ocr.ocr(text_image, cls=True)

            if not results or not results[0]:
                return [], 0.0

            # Parse with legal text parser
            gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
            parser = LegalTextParser(gray)
            segments = parser.parse_segments(results[0])

            # Calculate average confidence
            confidences = []
            for line in results[0]:
                if len(line[1]) > 1:
                    confidences.append(line[1][1])

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return segments, avg_conf

        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return [], 0.0

    def _segments_to_markdown(self, segments: List[TextSegment]) -> str:
        """
        Convert segments to markdown format.

        Args:
            segments: List of text/table segments

        Returns:
            Markdown formatted string
        """
        md_lines = []

        for seg in segments:
            if seg.category == 'table':
                md_lines.append(f"\n{seg.text}\n")
            else:
                # Build prefix
                prefix = ""

                # Heading tags
                if seg.tag == "h1":
                    prefix = "# "
                elif seg.tag == "h2":
                    prefix = "## "

                # Indentation
                if seg.is_indented:
                    prefix = "> " + prefix

                # Bold formatting
                text = seg.text
                if seg.is_bold:
                    text = f"**{text}**"

                md_lines.append(f"{prefix}{text}")

        return "\n\n".join(md_lines)

    def _segments_to_html(
        self,
        segments: List[TextSegment],
        page_number: int = 1
    ) -> str:
        """
        Convert segments to HTML format with proper styling.

        Args:
            segments: List of text/table segments
            page_number: Page number for title

        Returns:
            HTML formatted string
        """
        html_generator = HTMLGenerator()

        # Convert TextSegment to StyledText
        styled_segments = []

        for seg in segments:
            if seg.category == 'table':
                # Table: convert markdown table to HTML
                html_table = html_generator.markdown_table_to_html(seg.text)
                styled_segments.append(StyledText(
                    text=html_table,
                    style=TextStyle.TABLE,
                    y_pos=seg.y_pos,
                    is_bold=False
                ))
            else:
                # Determine style
                if seg.tag == "h1":
                    style = TextStyle.HEADING1
                elif seg.tag == "h2":
                    style = TextStyle.HEADING2
                elif seg.is_bold:
                    style = TextStyle.BOLD
                else:
                    style = TextStyle.NORMAL

                styled_segments.append(StyledText(
                    text=seg.text,
                    style=style,
                    y_pos=seg.y_pos,
                    is_bold=seg.is_bold
                ))

        # Generate HTML
        html = html_generator.generate_html(
            segments=styled_segments,
            title=f"OCR Document - Page {page_number}",
            include_css=True
        )

        return html

    def process_image(
        self,
        image_input: Union[str, np.ndarray, Path],
        page_number: int = 1
    ) -> OCRResult:
        """
        Process a single image through the OCR pipeline.

        Args:
            image_input: Image path or BGR image array
            page_number: Page number for tracking

        Returns:
            OCRResult with all extracted content
        """
        import time
        start_time = time.time()

        # Ensure initialized
        self.initialize()

        # Load image if path
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Failed to load image: {image_input}")
        else:
            image = image_input.copy()

        # Step 1: Layout analysis
        logger.debug(f"Page {page_number}: Analyzing layout...")
        tables, text_image = self.layout_analyzer.split_content(image)

        # Step 2: Process text
        logger.debug(f"Page {page_number}: Processing text...")
        text_segments, text_conf = self._process_text(text_image)

        # Step 3: Process tables
        logger.debug(f"Page {page_number}: Processing {len(tables)} tables...")
        table_segments = []
        for table in tables:
            table_text = self._process_table(table.image)
            table_segments.append(TextSegment(
                category="table",
                y_pos=table.box[1],
                text=table_text
            ))

        # Step 4: Merge and sort by Y position
        all_segments = text_segments + table_segments
        all_segments.sort(key=lambda x: x.y_pos)

        # Step 5: Generate HTML first (preserves formatting better)
        html = self._segments_to_html(all_segments, page_number)

        # Step 6: Generate Markdown from HTML (or direct conversion)
        markdown = self._segments_to_markdown(all_segments)

        # Calculate processing time
        processing_time = time.time() - start_time

        logger.info(
            f"Page {page_number} processed: "
            f"{len(text_segments)} text blocks, {len(tables)} tables, "
            f"{processing_time:.2f}s"
        )

        return OCRResult(
            page_number=page_number,
            raw_text="\n".join(s.text for s in all_segments),
            markdown=markdown,
            html=html,
            segments=all_segments,
            tables_count=len(tables),
            confidence=text_conf,
            processing_time=processing_time
        )

    def process_batch(
        self,
        images: List[Union[str, np.ndarray, Path]],
        start_page: int = 1
    ) -> List[OCRResult]:
        """
        Process multiple images in batch.

        Args:
            images: List of image paths or arrays
            start_page: Starting page number

        Returns:
            List of OCRResult objects
        """
        results = []

        for i, image in enumerate(images):
            page_num = start_page + i
            try:
                result = self.process_image(image, page_num)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                results.append(OCRResult(
                    page_number=page_num,
                    raw_text=f"[처리 실패: {str(e)}]",
                    markdown=f"[처리 실패: {str(e)}]",
                    confidence=0.0
                ))

        return results

    def cleanup(self) -> None:
        """Clean up GPU memory"""
        if self.got_model is not None:
            del self.got_model
            self.got_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("OCR Pipeline cleanup complete")


# Convenience function for single image processing
def process_single_image(
    image_path: str,
    got_model_path: str = "./weights/GOT-OCR2_0",
    yolo_model_path: str = "./weights/yolo_table_best.pt/yolov8n.pt"
) -> str:
    """
    Convenience function to process a single image.

    Args:
        image_path: Path to image file
        got_model_path: Path to GOT-OCR model
        yolo_model_path: Path to YOLO model

    Returns:
        Markdown formatted OCR result
    """
    pipeline = HybridOCRPipeline(got_model_path, yolo_model_path)
    result = pipeline.process_image(image_path)
    return result.markdown
