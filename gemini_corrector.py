"""
Legal Document OCR - Gemini AI Correction Module
=================================================
Final text correction using Google Gemini AI
Specialized for legal document terminology and formatting
"""

import os
import re
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("google-generativeai not installed. Gemini correction unavailable.")
    GEMINI_AVAILABLE = False

from config import get_settings


@dataclass
class CorrectionResult:
    """Result of Gemini correction"""
    original_text: str
    corrected_text: str
    corrections_made: int
    confidence: float
    processing_time: float


class GeminiCorrector:
    """
    Legal document text corrector using Google Gemini AI.

    Features:
    - Legal terminology correction
    - OCR error fixing (similar character confusion)
    - Table structure validation
    - Korean legal document specialized
    - Rate limiting and retry logic
    """

    # System prompt for legal document correction
    SYSTEM_PROMPT = """당신은 한국 법률 문서 전문 교정사입니다.
OCR로 인식된 법률 문서의 오류를 교정해주세요.

교정 규칙:
1. OCR 인식 오류 수정 (예: 0과 O, 1과 l, 乙과 Z 혼동)
2. 법률 용어 정확성 확인 (예: 갑, 을, 병, 정 / 甲, 乙, 丙, 丁)
3. 조문 번호 형식 통일 (예: 제1조, 제2항, 제3호)
4. 날짜 형식 확인 (예: 2024. 1. 15.)
5. 금액 형식 확인 (예: 금 50,000,000원)
6. 표의 구조 유지 (마크다운 표 형식 보존)
7. 띄어쓰기 및 맞춤법 교정

중요: 원문의 의미를 변경하지 마세요. 명백한 OCR 오류만 수정하세요.
교정된 텍스트만 출력하고, 설명은 포함하지 마세요."""

    # Common OCR error patterns for Korean legal documents
    OCR_ERROR_PATTERNS = [
        # 갑을병정 관련
        (r'\bZ\b(?=\s*[이가는을의])', '乙'),
        (r'\bE\b(?=\s*[이가는을의])', '乙'),
        (r'갑\s*Z', '갑 乙'),
        (r'甲\s*Z', '甲 乙'),

        # 숫자/문자 혼동
        (r'제(\d+)[oO]조', r'제\g<1>0조'),
        (r'제[lI](\d+)조', r'제1\g<1>조'),

        # 법조문 형식
        (r'제\s*(\d+)\s*조', r'제\g<1>조'),
        (r'제\s*(\d+)\s*항', r'제\g<1>항'),
        (r'제\s*(\d+)\s*호', r'제\g<1>호'),

        # 금액 표기
        (r'금\s+(\d)', r'금 \g<1>'),
        (r'(\d),(\d{3}),(\d{3})원', r'\g<1>,\g<2>,\g<3>원'),

        # 날짜 표기
        (r'(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})', r'\g<1>. \g<2>. \g<3>.'),
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Gemini corrector.

        Args:
            api_key: Gemini API key (or from env GEMINI_API_KEY)
            model_name: Gemini model to use
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model = None

        # Get API key
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")

        if self.api_key and GEMINI_AVAILABLE:
            self._initialize_model()
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini SDK not installed")
            if not self.api_key:
                logger.warning("Gemini API key not provided")

    def _initialize_model(self) -> None:
        """Initialize Gemini model"""
        try:
            genai.configure(api_key=self.api_key)

            # Configure generation parameters
            generation_config = {
                "temperature": 0.1,  # Low temperature for consistent correction
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }

            # Safety settings - allow legal content
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=self.SYSTEM_PROMPT
            )

            logger.info(f"Gemini model initialized: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None

    def _apply_regex_corrections(self, text: str) -> Tuple[str, int]:
        """
        Apply regex-based corrections before Gemini.

        Args:
            text: Original text

        Returns:
            Tuple of (corrected text, number of corrections)
        """
        corrected = text
        corrections = 0

        for pattern, replacement in self.OCR_ERROR_PATTERNS:
            new_text = re.sub(pattern, replacement, corrected)
            if new_text != corrected:
                corrections += 1
                corrected = new_text

        return corrected, corrections

    def _split_into_chunks(self, text: str, max_chars: int = 4000) -> List[str]:
        """
        Split text into chunks for processing.

        Args:
            text: Text to split
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def correct_with_gemini(self, text: str) -> str:
        """
        Correct text using Gemini AI.

        Args:
            text: Text to correct

        Returns:
            Corrected text
        """
        if self.model is None:
            logger.warning("Gemini model not available, returning original text")
            return text

        prompt = f"""다음 법률 문서의 OCR 오류를 교정해주세요.
원문의 구조와 형식을 유지하면서 명백한 오류만 수정하세요.

---
{text}
---

교정된 텍스트:"""

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)

                if response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")

            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        # Return original if all retries failed
        return text

    def correct(
        self,
        text: str,
        use_gemini: bool = True
    ) -> CorrectionResult:
        """
        Full correction pipeline.

        Args:
            text: Text to correct
            use_gemini: Whether to use Gemini AI (in addition to regex)

        Returns:
            CorrectionResult with corrected text and metadata
        """
        start_time = time.time()

        # Step 1: Apply regex corrections
        corrected, regex_corrections = self._apply_regex_corrections(text)

        # Step 2: Apply Gemini corrections if enabled
        if use_gemini and self.model is not None:
            chunks = self._split_into_chunks(corrected)
            corrected_chunks = []

            for chunk in chunks:
                corrected_chunk = self.correct_with_gemini(chunk)
                corrected_chunks.append(corrected_chunk)

            corrected = "\n\n".join(corrected_chunks)

        # Count total corrections (approximate)
        total_corrections = regex_corrections
        if use_gemini and corrected != text:
            # Rough estimate based on edit distance
            total_corrections += sum(1 for a, b in zip(text, corrected) if a != b) // 5

        processing_time = time.time() - start_time

        return CorrectionResult(
            original_text=text,
            corrected_text=corrected,
            corrections_made=total_corrections,
            confidence=0.95 if use_gemini else 0.8,
            processing_time=processing_time
        )

    def correct_batch(
        self,
        texts: List[str],
        use_gemini: bool = True
    ) -> List[CorrectionResult]:
        """
        Correct multiple texts.

        Args:
            texts: List of texts to correct
            use_gemini: Whether to use Gemini AI

        Returns:
            List of CorrectionResult objects
        """
        results = []

        for i, text in enumerate(texts):
            logger.debug(f"Correcting text {i + 1}/{len(texts)}")
            result = self.correct(text, use_gemini)
            results.append(result)

        return results


# Convenience function
def correct_legal_text(
    text: str,
    api_key: Optional[str] = None,
    use_gemini: bool = True
) -> str:
    """
    Convenience function to correct legal document text.

    Args:
        text: Text to correct
        api_key: Gemini API key (optional)
        use_gemini: Whether to use Gemini AI

    Returns:
        Corrected text
    """
    corrector = GeminiCorrector(api_key=api_key)
    result = corrector.correct(text, use_gemini=use_gemini)
    return result.corrected_text


# Validate table format
def validate_markdown_table(table_text: str) -> Tuple[bool, str]:
    """
    Validate and fix markdown table format.

    Args:
        table_text: Markdown table text

    Returns:
        Tuple of (is_valid, fixed_table)
    """
    lines = table_text.strip().split('\n')

    if len(lines) < 2:
        return False, table_text

    # Check for pipe characters
    if not all('|' in line for line in lines):
        return False, table_text

    # Fix common issues
    fixed_lines = []
    max_cols = 0

    for line in lines:
        # Ensure leading/trailing pipes
        line = line.strip()
        if not line.startswith('|'):
            line = '|' + line
        if not line.endswith('|'):
            line = line + '|'

        cols = len(line.split('|')) - 2  # Subtract empty strings at ends
        max_cols = max(max_cols, cols)
        fixed_lines.append(line)

    # Normalize column count
    normalized = []
    for line in fixed_lines:
        parts = line.split('|')
        while len(parts) - 2 < max_cols:
            parts.insert(-1, ' ')
        normalized.append('|'.join(parts))

    # Ensure separator row exists
    if len(normalized) >= 2:
        sep_idx = 1
        if not re.match(r'\|\s*[-:]+\s*\|', normalized[1]):
            sep_row = '|' + '|'.join(['---'] * max_cols) + '|'
            normalized.insert(1, sep_row)

    fixed_table = '\n'.join(normalized)
    return True, fixed_table
