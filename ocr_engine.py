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
import time
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

from layout_analyzer import LayoutAnalyzer, DetectedRegion, TextFormatInfo, InvisibleTableInfo
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
    alignment: str = "left"  # left, center, right
    line_spacing_before: int = 0  # Pixels of empty space before this text
    x_pos: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    font_size: float = 12.0


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
    - Legal-specific typo correction (Upstage OCR 버그 보완)
    """

    # Legal document style thresholds
    H1_SCALE = 1.4       # 1.4x median height = h1
    H2_SCALE = 1.15      # 1.15x median height = h2
    INDENT_PX = 20       # 20px from baseline = indented
    BOLD_RATIO = 1.10    # 10% denser = bold

    # ============================================
    # 갑을병정(甲乙丙丁) 한자 오인 교정
    # ============================================
    PARTY_CORRECTIONS = {
        # 을(乙) 오인 패턴
        'Z': '乙',
        'z': '乙',
        'E': '乙',      # E가 乙로 오인되는 경우
        '2': '乙',      # 숫자 2가 乙로 오인되는 경우 (컨텍스트 확인 필요)
        # 병(丙) 오인 패턴
        'C': '丙',      # C가 丙으로 오인
        # 정(丁) 오인 패턴
        'T': '丁',      # T가 丁으로 오인
        # 무(戊), 기(己), 경(庚), 신(辛), 임(壬), 계(癸)
    }

    # 갑을병정 시퀀스에서의 교정
    PARTY_SEQUENCE_PATTERNS = [
        # 갑 다음에 오는 Z, E, 2 → 을
        (r'(갑|甲)\s*[,、]\s*[ZzE2](?=\s*[,、]|\s*$|\s+)', r'\1, 乙'),
        (r'(갑|甲)\s+[ZzE2](?=\s|$)', r'\1 乙'),
        # 을 다음에 오는 C → 병
        (r'(을|乙)\s*[,、]\s*[Cc](?=\s*[,、]|\s*$|\s+)', r'\1, 丙'),
        (r'(을|乙)\s+[Cc](?=\s|$)', r'\1 丙'),
        # 병 다음에 오는 T → 정
        (r'(병|丙)\s*[,、]\s*[Tt](?=\s*[,、]|\s*$|\s+)', r'\1, 丁'),
        (r'(병|丙)\s+[Tt](?=\s|$)', r'\1 丁'),
        # 단독 패턴: "갑과 Z" → "갑과 乙"
        (r'(갑|甲)\s*(과|와|및|,)\s*[ZzE2]', r'\1\2 乙'),
        (r'[ZzE2]\s*(과|와|및|,)\s*(갑|甲)', r'乙\1 \2'),
    ]

    # ============================================
    # 숫자/문자 혼동 교정 (0과 o, 1과 l/I 등)
    # ============================================
    NUMBER_LETTER_PATTERNS = [
        # 법조문에서 o/O를 0으로 교정 (제1o조 → 제10조)
        (r'(제\s*\d+)[oO](\d*\s*조)', r'\g<1>0\g<2>'),
        (r'(제\s*\d+\s*조\s*제?\s*\d*)[oO](\d*\s*항)', r'\g<1>0\g<2>'),
        (r'(제\s*\d+\s*조\s*제?\s*\d*\s*항?\s*제?\s*\d*)[oO](\d*\s*호)', r'\g<1>0\g<2>'),
        # 금액에서 o/O를 0으로 교정 (1oo,ooo원 → 100,000원)
        (r'(\d)[oO](\d)', r'\g<1>0\g<2>'),
        (r'(\d)[oO]([,\.])', r'\g<1>0\g<2>'),
        (r'([,\.])[oO](\d)', r'\g<1>0\g<2>'),
        # 연도에서 o를 0으로 (2o23년 → 2023년)
        (r'(19|20)\s*[oO]\s*(\d)\s*년', r'\g<1>0\g<2>년'),
        (r'(19|20)\s*(\d)\s*[oO]\s*년', r'\g<1>\g<2>0년'),
        # l/I를 1로 교정 (조문 번호에서)
        (r'(제\s*)[lI](\d*\s*조)', r'\g<1>1\g<2>'),
        (r'(제\s*\d+)[lI](\s*조)', r'\g<1>1\g<2>'),
        # 날짜에서 l을 1로 (l2월 → 12월, 3l일 → 31일)
        (r'([lI])(\d\s*월)', r'1\g<2>'),
        (r'(\d)[lI](\s*월)', r'\g<1>1\g<2>'),
        (r'([lI])(\d\s*일)', r'1\g<2>'),
        (r'(\d)[lI](\s*일)', r'\g<1>1\g<2>'),
    ]

    # ============================================
    # 로마 숫자 교정
    # ============================================
    ROMAN_NUMERALS = {
        'I': 'Ⅰ', 'II': 'Ⅱ', 'III': 'Ⅲ',
        'IV': 'Ⅳ', 'V': 'Ⅴ', 'VI': 'Ⅵ',
        'VII': 'Ⅶ', 'VIII': 'Ⅷ', 'IX': 'Ⅸ', 'X': 'Ⅹ',
        'XI': 'Ⅺ', 'XII': 'Ⅻ'
    }

    # 로마 숫자 패턴 (문장/항목 시작 부분에서)
    ROMAN_NUMERAL_PATTERNS = [
        # 항목 번호로 사용된 로마 숫자 (I. II. III. 등)
        (r'^([IⅠ])\.(\s)', r'Ⅰ.\g<2>'),
        (r'^(II|[IⅠ]{2})\.(\s)', r'Ⅱ.\g<2>'),
        (r'^(III|[IⅠ]{3})\.(\s)', r'Ⅲ.\g<2>'),
        (r'^(IV)\.(\s)', r'Ⅳ.\g<2>'),
        (r'^([VⅤ])\.(\s)', r'Ⅴ.\g<2>'),
        # 괄호 안 로마 숫자
        (r'\(([IⅠ])\)', r'(Ⅰ)'),
        (r'\((II|[IⅠ]{2})\)', r'(Ⅱ)'),
        (r'\((III|[IⅠ]{3})\)', r'(Ⅲ)'),
        (r'\((IV)\)', r'(Ⅳ)'),
        (r'\(([VⅤ])\)', r'(Ⅴ)'),
    ]

    # ============================================
    # 법률 용어 특수 교정
    # ============================================
    LEGAL_TERM_PATTERNS = [
        # 조항호 띄어쓰기 정규화
        (r'제\s+(\d+)\s*조', r'제\g<1>조'),
        (r'제\s*(\d+)\s+조', r'제\g<1>조'),
        (r'제\s+(\d+)\s*항', r'제\g<1>항'),
        (r'제\s*(\d+)\s+항', r'제\g<1>항'),
        (r'제\s+(\d+)\s*호', r'제\g<1>호'),
        (r'제\s*(\d+)\s+호', r'제\g<1>호'),
        # 원고/피고 오타
        (r'원고(?![가-힣])', '원고'),
        (r'피고(?![가-힣])', '피고'),
        # 금액 단위 교정
        (r'(\d)\s*원(?!\s*고|\s*피)', r'\g<1>원'),
        # 날짜 형식 정규화
        (r'(\d{4})\s*[.·]\s*(\d{1,2})\s*[.·]\s*(\d{1,2})', r'\g<1>. \g<2>. \g<3>.'),
    ]

    # ============================================
    # 특수문자/기호 교정
    # ============================================
    SYMBOL_CORRECTIONS = {
        '「': '「',  # 꺾쇠 정규화
        '」': '」',
        '『': '『',
        '』': '』',
        '"': '"',   # 따옴표 정규화
        '"': '"',
        ''': "'",
        ''': "'",
        '−': '-',   # 하이픈 정규화
        '–': '-',
        '—': '-',
        '．': '.',  # 마침표 정규화
        '，': ',',  # 쉼표 정규화
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
        Comprehensive OCR error correction for Korean legal documents.

        Corrections include:
        - 갑을병정(甲乙丙丁) 한자 오인 (Z→乙, E→乙 등)
        - 숫자/문자 혼동 (0↔o, 1↔l/I)
        - 로마 숫자 정규화 (I→Ⅰ, II→Ⅱ 등)
        - 법률 용어 띄어쓰기 정규화
        - 특수문자 정규화

        Args:
            text: Raw OCR text

        Returns:
            Corrected text
        """
        if not text:
            return text

        # Step 1: 특수문자/기호 정규화
        for old_char, new_char in self.SYMBOL_CORRECTIONS.items():
            text = text.replace(old_char, new_char)

        # Step 2: 갑을병정 시퀀스 교정 (컨텍스트 기반)
        # 갑, 甲이 포함된 경우에만 을병정 교정 적용
        if any(party in text for party in ['갑', '甲', '을', '乙', '병', '丙']):
            for pattern, replacement in self.PARTY_SEQUENCE_PATTERNS:
                text = re.sub(pattern, replacement, text)

        # Step 3: 숫자/문자 혼동 교정 (o↔0, l↔1 등)
        for pattern, replacement in self.NUMBER_LETTER_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # Step 4: 로마 숫자 교정
        for pattern, replacement in self.ROMAN_NUMERAL_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        # Step 5: 법률 용어 띄어쓰기 정규화
        for pattern, replacement in self.LEGAL_TERM_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # Step 6: 단독 로마 숫자 변환 (문맥상 항목 번호로 판단되는 경우)
        words = text.split()
        corrected_words = []
        for i, word in enumerate(words):
            # 마침표가 붙은 로마 숫자 (I. II. III. 등)
            if word.rstrip('.') in self.ROMAN_NUMERALS and word.endswith('.'):
                corrected_words.append(self.ROMAN_NUMERALS[word.rstrip('.')] + '.')
            # 괄호 없이 단독으로 쓰인 로마 숫자 (문장 시작 또는 이전 단어가 조사/접속사)
            elif word in self.ROMAN_NUMERALS:
                # 문장 시작이거나 이전 단어가 특정 조사인 경우에만 변환
                if i == 0 or (i > 0 and corrected_words[-1] in [',', '및', '또는', '그리고']):
                    corrected_words.append(self.ROMAN_NUMERALS[word])
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def parse_segments(
        self,
        ocr_result: List,
        page_width: int = 0
    ) -> List[TextSegment]:
        """
        Parse PaddleOCR results into structured segments with alignment and spacing.

        Args:
            ocr_result: PaddleOCR result list
            page_width: Width of the page for alignment detection

        Returns:
            List of TextSegment objects with alignment and spacing info
        """
        if not ocr_result:
            return []

        # Collect statistics
        heights = []
        densities = []
        x_positions = []
        bboxes = []

        for line in ocr_result:
            box = line[0]
            h, x, y, w, pts = self.get_geometry(box)
            heights.append(h)
            x_positions.append(x)
            bboxes.append((x, y, x + w, y + h))

            density = self.calculate_density(pts)
            if density > 0:
                densities.append(density)

        # Calculate medians
        median_h = statistics.median(heights) if heights else 20
        min_x = min(x_positions) if x_positions else 0
        max_x = max(x_positions) if x_positions else 0
        median_d = statistics.median(densities) if densities else 0.5

        # If page_width not provided, estimate from bboxes
        if page_width == 0 and bboxes:
            page_width = max(b[2] for b in bboxes) + min_x

        # Sort indices by Y position for line spacing calculation
        sorted_indices = sorted(range(len(ocr_result)), key=lambda i: bboxes[i][1])

        # Parse each line
        segments = []
        for idx, line in enumerate(ocr_result):
            box = line[0]
            raw_text = line[1][0]
            confidence = line[1][1] if len(line[1]) > 1 else 0.0

            h, x, y, w, pts = self.get_geometry(box)
            bbox = (x, y, x + w, y + h)

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

            # Detect alignment
            alignment = self._detect_alignment(bbox, page_width)

            # Calculate line spacing before this text
            line_spacing = self._calculate_line_spacing(idx, bboxes, sorted_indices)

            segments.append(TextSegment(
                category="text",
                y_pos=y,
                text=text,
                tag=tag,
                is_bold=is_bold,
                is_indented=is_indented,
                alignment=alignment,
                line_spacing_before=line_spacing,
                x_pos=x,
                bbox=bbox,
                font_size=h * 0.75  # Approximate point size
            ))

        return segments

    def _detect_alignment(
        self,
        bbox: Tuple[int, int, int, int],
        page_width: int
    ) -> str:
        """
        Detect text alignment based on position.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            page_width: Page width

        Returns:
            Alignment string ("left", "center", "right")
        """
        if page_width == 0:
            return "left"

        x1, y1, x2, y2 = bbox
        text_width = x2 - x1
        text_center = (x1 + x2) / 2
        page_center = page_width / 2

        # Full-width text is left-aligned
        if text_width > page_width * 0.7:
            return "left"

        # Center tolerance (within 10% of center)
        center_tolerance = page_width * 0.10

        # Check for center alignment
        if abs(text_center - page_center) < center_tolerance:
            left_space = x1
            right_space = page_width - x2
            if abs(left_space - right_space) < page_width * 0.15:
                return "center"

        # Check for right alignment
        right_margin = page_width * 0.92
        if x2 > right_margin and x1 > page_width * 0.4:
            return "right"

        return "left"

    def _calculate_line_spacing(
        self,
        current_idx: int,
        bboxes: List[Tuple[int, int, int, int]],
        sorted_indices: List[int]
    ) -> int:
        """
        Calculate vertical spacing before a text block.

        Args:
            current_idx: Index of current bbox
            bboxes: List of all bboxes
            sorted_indices: Indices sorted by Y position

        Returns:
            Pixel spacing before this text block
        """
        current_pos = sorted_indices.index(current_idx)
        if current_pos == 0:
            return 0

        prev_idx = sorted_indices[current_pos - 1]
        prev_bbox = bboxes[prev_idx]
        current_bbox = bboxes[current_idx]

        # Calculate gap between previous block's bottom and current block's top
        gap = current_bbox[1] - prev_bbox[3]

        return max(0, gap)


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
    ) -> Tuple[List[TextSegment], float, int]:
        """
        Process text image with PaddleOCR.

        Args:
            text_image: BGR image with text (tables removed)

        Returns:
            Tuple of (parsed segments, average confidence, page_width)
        """
        if self.paddle_ocr is None:
            return [], 0.0, 0

        try:
            results = self.paddle_ocr.ocr(text_image, cls=True)

            if not results or not results[0]:
                return [], 0.0, text_image.shape[1] if len(text_image.shape) >= 2 else 0

            # Get page width from image
            page_width = text_image.shape[1] if len(text_image.shape) >= 2 else 0

            # Parse with legal text parser
            gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
            parser = LegalTextParser(gray)
            segments = parser.parse_segments(results[0], page_width)

            # Calculate average confidence
            confidences = []
            for line in results[0]:
                if len(line[1]) > 1:
                    confidences.append(line[1][1])

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return segments, avg_conf, page_width

        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return [], 0.0, 0

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
        page_number: int = 1,
        page_width: int = 0
    ) -> str:
        """
        Convert segments to HTML format with proper styling, alignment, and spacing.

        Args:
            segments: List of text/table segments
            page_number: Page number for title
            page_width: Page width for relative positioning

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
                    is_bold=False,
                    alignment=seg.alignment,
                    line_spacing_before=seg.line_spacing_before
                ))
            elif seg.category == 'invisible_table':
                # Invisible table (government forms)
                html_table = html_generator.markdown_table_to_html(seg.text)
                styled_segments.append(StyledText(
                    text=html_table,
                    style=TextStyle.INVISIBLE_TABLE,
                    y_pos=seg.y_pos,
                    is_bold=False,
                    alignment=seg.alignment,
                    line_spacing_before=seg.line_spacing_before
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
                    x_pos=seg.x_pos,
                    font_size=seg.font_size,
                    is_bold=seg.is_bold,
                    alignment=seg.alignment,
                    line_spacing_before=seg.line_spacing_before,
                    page_width=page_width
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

        Includes:
        - Text alignment detection (left, center, right)
        - Line spacing preservation
        - Invisible table detection for government forms
        - Bold/heading detection
        - Font size estimation

        Args:
            image_input: Image path or BGR image array
            page_number: Page number for tracking

        Returns:
            OCRResult with all extracted content
        """
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

        # Get page dimensions
        page_height, page_width = image.shape[:2]

        # Step 1: Layout analysis
        logger.debug(f"Page {page_number}: Analyzing layout...")
        tables, text_image = self.layout_analyzer.split_content(image)

        # Step 2: Process text (returns segments, confidence, page_width)
        logger.debug(f"Page {page_number}: Processing text...")
        text_segments, text_conf, detected_width = self._process_text(text_image)

        # Use detected width if available, otherwise use image width
        effective_width = detected_width if detected_width > 0 else page_width

        # Step 3: Process tables
        logger.debug(f"Page {page_number}: Processing {len(tables)} tables...")
        table_segments = []
        for table in tables:
            table_text = self._process_table(table.image)
            # Detect if this is a borderless/invisible table (heuristic_table)
            is_invisible = table.region_type == "heuristic_table"
            table_segments.append(TextSegment(
                category="invisible_table" if is_invisible else "table",
                y_pos=table.box[1],
                text=table_text,
                bbox=table.box
            ))

        # Step 4: Merge and sort by Y position
        all_segments = text_segments + table_segments
        all_segments.sort(key=lambda x: x.y_pos)

        # Step 5: Generate HTML first (preserves formatting better)
        html = self._segments_to_html(all_segments, page_number, effective_width)

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
