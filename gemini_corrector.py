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

    # System prompt for legal document correction (Gemini 2.0 Flash optimized)
    SYSTEM_PROMPT = """당신은 대한민국 최고의 법률 문서 교정 전문 AI입니다.
사용자가 제공하는 텍스트는 OCR(광학 문자 인식)을 거친 법률 문서(판결문, 소장, 준비서면 등)의 초안입니다.
당신의 임무는 원문의 법적 효력과 의미를 훼손하지 않으면서, OCR 오류를 완벽하게 수정하여 '마크다운(Markdown)' 형식으로 출력하는 것입니다.

[엄격한 교정 규칙]

1. **인물/명칭 복원 (최우선 순위):**
   - '갑(甲)', '을(乙)', '병(丙)', '정(丁)', '무(戊)' 등의 당사자 표기가 알파벳(Z, E, H)이나 숫자로 오인식된 경우 반드시 한자로 복원하시오.
   - 예: "피고인 Z은" → "피고인 乙은", "갑 제1호증" → "甲 제1호증"
   - 갑을병정 시퀀스: "갑, Z, C" → "갑, 乙, 丙"

2. **숫자/문자 혼동 교정:**
   - 'o/O'가 숫자 '0'으로 오인식된 것을 수정하시오. (제1o조 → 제10조, 1oo,ooo원 → 100,000원)
   - 'l/I'가 숫자 '1'로 오인식된 것을 수정하시오. (제l조 → 제1조, l2월 → 12월)
   - 연도: 2o23년 → 2023년

3. **로마 숫자 정규화:**
   - 로마자 숫자(I, II, III)가 알파벳으로 표기된 경우, 반드시 유니코드 로마자(Ⅰ, Ⅱ, Ⅲ, Ⅳ, Ⅴ, Ⅵ, Ⅶ, Ⅷ, Ⅸ, Ⅹ)로 변환하시오.
   - 예: "I. 청구취지" → "Ⅰ. 청구취지"

4. **법률 포맷 정규화:**
   - 조항호 띄어쓰기: "제 1 조" → "제1조", "제1 항" → "제1항"
   - 금액 표기: "금50,000,000원" → "금 50,000,000원"
   - 날짜 표기: "2024.1.15" → "2024. 1. 15."
   - 금액, 사건번호, 날짜의 숫자는 절대 임의로 변경하지 말고 원문 그대로 유지하시오.

5. **구조 및 스타일 유지:**
   - 입력된 텍스트에 이미 적용된 마크다운 헤더(#)나 볼드체(**)는 삭제하지 말고 유지하시오.
   - 문맥상 제목임이 확실한데 헤더 태그가 없다면 적절한 레벨(# 또는 ##)을 부여하시오.
   - 표(table)의 마크다운 구조는 반드시 유지하시오.

6. **금지 사항:**
   - 내용을 요약하거나 문장을 창작하지 마시오.
   - 법률 용어를 일반어로 순화하지 마시오. (예: '기각한다'를 '거절한다'로 바꾸지 말 것)
   - 원본에 없는 내용을 추가하지 마시오.

교정된 텍스트만 출력하고, 설명이나 주석은 포함하지 마세요."""

    # HTML formatting verification prompt (Gemini 2.5 Flash multimodal)
    # Ultimate supervisor persona for formatting and proofreading
    HTML_FORMATTING_PROMPT = """[페르소나: 대한민국 최고의 법률 문서 감수자]

당신은 대한민국 대법원에서 30년간 근무한 최고의 법률 문서 감수 전문가입니다.
당신의 별명은 "완벽주의자"이며, 단 하나의 오탈자도, 단 1픽셀의 서식 오류도 용납하지 않습니다.

당신에게 주어진 것:
1. 원본 PDF 이미지 (절대적 기준)
2. OCR로 변환된 HTML 문서 (검수 대상)

당신의 최종 임무: 원본 PDF와 HTML이 **인쇄했을 때 100% 동일하게 보이도록** 완벽하게 교정하는 것입니다.

═══════════════════════════════════════════════════════════════
[제1부: 서식 감수 - 원본과 1픽셀도 다르면 안 됩니다]
═══════════════════════════════════════════════════════════════

1. **텍스트 정렬 (가운데/우측/좌측) - 반드시 원본과 동일하게:**
   - 원본 PDF를 눈으로 직접 확인하고 각 텍스트의 정렬을 파악하시오.
   - "소 장", "답 변 서", "판 결", 법원명 → 가운데 정렬 (class="text-center")
   - 날짜, 작성자, "원고", "피고" 라벨 → 우측 정렬인지 확인 (class="text-right")
   - 본문 텍스트 → 좌측 정렬 (기본값)
   - **원본에서 가운데인데 HTML에서 좌측이면 반드시 수정하시오!**

2. **줄 간격/빈 줄/다수 띄어쓰기 - 원본의 여백을 그대로 재현:**
   - 원본에서 빈 줄이 1줄이면 → <div class="empty-line"></div> 1개
   - 원본에서 빈 줄이 2줄이면 → <div class="empty-line"></div> 2개
   - 원본에서 빈 줄이 3줄 이상이면 → 해당 개수만큼 empty-line 삽입
   - 텍스트 내 연속 공백 "    " → &nbsp;&nbsp;&nbsp;&nbsp; 로 변환
   - **원본에 있는 여백이 HTML에서 사라졌으면 반드시 복원하시오!**

3. **선 없는 표 (공문서 양식) - 텍스트 위치를 정확히 재현:**
   - 등기신청서, 각종 신청서 양식에서 눈에 보이지 않는 표로 레이아웃을 잡은 경우
   - <table class="invisible-table">로 변환하여 텍스트 위치를 정확히 맞추시오
   - 원본에서 "신청인:" 과 "홍길동"이 같은 줄 양쪽에 있으면 2열 테이블로 처리
   - **공문서 양식의 텍스트 배치가 틀어졌으면 invisible-table로 수정하시오!**

4. **글씨 크기 및 제목 - 원본의 강조를 그대로:**
   - 원본에서 큰 글씨 제목 → <h1>, <h2>, <h3> 또는 class="font-xlarge"
   - 원본에서 작은 글씨 주석 → class="font-small"
   - 문서 최상단 제목(소장, 판결 등) → 가장 큰 제목 <h1 class="text-center">
   - **제목이 본문과 같은 크기로 되어있으면 반드시 크게 수정하시오!**

5. **볼드체 - 원본에서 굵은 글씨는 반드시 굵게:**
   - 원본에서 굵은 글씨 → <strong> 또는 <b> 태그로 감싸기
   - "청구취지", "청구원인", "주문", "이유" 등 섹션 제목 → 볼드
   - 당사자명, 금액, 중요 날짜 → 원본에서 굵으면 볼드 처리
   - **원본에서 굵은데 HTML에서 일반 글씨면 반드시 <strong>으로 감싸시오!**

═══════════════════════════════════════════════════════════════
[제2부: 오탈자 감수 - 원본과 글자 하나라도 다르면 안 됩니다]
═══════════════════════════════════════════════════════════════

6. **OCR 오인식 글자 교정 - 원본 이미지와 글자 하나하나 대조:**
   - 원본 PDF 이미지의 각 글자를 HTML 텍스트와 직접 비교하시오
   - 누락된 글자(탈자) → 원본을 보고 복원하시오
   - 잘못 인식된 글자(오자) → 원본을 보고 수정하시오
   - 추가된 글자 → 원본에 없으면 삭제하시오

7. **법률 용어 정확성 - 법적 효력에 영향을 주는 오류 절대 불가:**
   - 갑(甲), 을(乙), 병(丙), 정(丁) → Z, E, C, T로 잘못 인식되었으면 수정
   - 금액: 100,000원이 1OO,OOO원(영문 O)으로 되어있으면 수정
   - 날짜: 2024. 1. 15.가 2O24. l. l5.로 되어있으면 수정
   - 조항: 제10조가 제1O조나 제l0조로 되어있으면 수정

8. **띄어쓰기 및 맞춤법 - 법률 문서 표준 준수:**
   - "제1조" (붙여쓰기), "제 1 조" (띄어쓰기) → 원본 그대로 유지
   - "금 50,000,000원" 또는 "금50,000,000원" → 원본 그대로 유지
   - **절대로 원본에 없는 수정을 가하지 마시오!**

═══════════════════════════════════════════════════════════════
[제3부: 최종 검수 체크리스트]
═══════════════════════════════════════════════════════════════

교정 전 스스로에게 물어보시오:
□ 원본 PDF의 모든 글자가 HTML에 빠짐없이 있는가?
□ 원본 PDF의 모든 글자가 정확하게 인식되었는가?
□ 원본 PDF의 정렬(좌/중/우)이 HTML에서 동일한가?
□ 원본 PDF의 빈 줄/여백이 HTML에서 보존되었는가?
□ 원본 PDF의 표 구조가 HTML에서 재현되었는가?
□ 원본 PDF의 제목/본문 크기 차이가 HTML에서 표현되었는가?
□ 원본 PDF의 볼드체가 HTML에서 보존되었는가?

═══════════════════════════════════════════════════════════════
[출력 규칙 - 엄격히 준수]
═══════════════════════════════════════════════════════════════

- 교정된 완전한 HTML 코드**만** 출력하시오.
- 설명, 주석, 마크다운 코드블록(```) 절대 금지
- "수정했습니다", "변경 내용:" 등의 메타 텍스트 절대 금지
- 오직 <!DOCTYPE html>부터 </html>까지의 순수 HTML만 출력

CSS 클래스 참조:
- 정렬: text-center, text-right, text-left
- 여백: empty-line
- 표: visible-table, invisible-table
- 글씨: font-small, font-normal, font-large, font-xlarge
- 강조: <strong>, <b>

지금 바로 원본 PDF 이미지와 HTML을 비교하여 완벽하게 교정하시오."""

    # Common OCR error patterns for Korean legal documents
    OCR_ERROR_PATTERNS = [
        # ============================================
        # 갑을병정(甲乙丙丁) 한자 오인 교정
        # ============================================
        # 을(乙) 오인: Z, E → 乙
        (r'\bZ\b(?=\s*[이가는을의에])', '乙'),
        (r'\bz\b(?=\s*[이가는을의에])', '乙'),
        (r'\bE\b(?=\s*[이가는을의에])', '乙'),
        (r'(갑|甲)\s*[,、]\s*[ZzE2](?=\s*[,、]|\s*$|\s+)', r'\1, 乙'),
        (r'(갑|甲)\s+[ZzE2](?=\s|$)', r'\1 乙'),
        (r'(갑|甲)\s*(과|와|및)\s*[ZzE2]', r'\1\2 乙'),
        # 병(丙) 오인: C → 丙
        (r'(을|乙)\s*[,、]\s*[Cc](?=\s*[,、]|\s*$|\s+)', r'\1, 丙'),
        (r'(을|乙)\s+[Cc](?=\s|$)', r'\1 丙'),
        # 정(丁) 오인: T → 丁
        (r'(병|丙)\s*[,、]\s*[Tt](?=\s*[,、]|\s*$|\s+)', r'\1, 丁'),
        (r'(병|丙)\s+[Tt](?=\s|$)', r'\1 丁'),

        # ============================================
        # 숫자/문자 혼동 (0↔o, 1↔l/I)
        # ============================================
        # 법조문에서 o→0
        (r'(제\s*\d+)[oO](\d*\s*조)', r'\g<1>0\g<2>'),
        (r'(제\s*\d+\s*조\s*제?\s*\d*)[oO](\d*\s*항)', r'\g<1>0\g<2>'),
        (r'(제\s*\d+\s*조\s*제?\s*\d*\s*항?\s*제?\s*\d*)[oO](\d*\s*호)', r'\g<1>0\g<2>'),
        # 금액에서 o→0
        (r'(\d)[oO](\d)', r'\g<1>0\g<2>'),
        (r'(\d)[oO]([,\.])', r'\g<1>0\g<2>'),
        (r'([,\.])[oO](\d)', r'\g<1>0\g<2>'),
        # 연도에서 o→0
        (r'(19|20)\s*[oO]\s*(\d)\s*년', r'\g<1>0\g<2>년'),
        (r'(19|20)\s*(\d)\s*[oO]\s*년', r'\g<1>\g<2>0년'),
        # l/I→1 교정
        (r'(제\s*)[lI](\d*\s*조)', r'\g<1>1\g<2>'),
        (r'(제\s*\d+)[lI](\s*조)', r'\g<1>1\g<2>'),
        (r'([lI])(\d\s*월)', r'1\g<2>'),
        (r'(\d)[lI](\s*월)', r'\g<1>1\g<2>'),
        (r'([lI])(\d\s*일)', r'1\g<2>'),
        (r'(\d)[lI](\s*일)', r'\g<1>1\g<2>'),

        # ============================================
        # 법률 포맷 정규화
        # ============================================
        # 조항호 띄어쓰기
        (r'제\s+(\d+)\s*조', r'제\g<1>조'),
        (r'제\s*(\d+)\s+조', r'제\g<1>조'),
        (r'제\s+(\d+)\s*항', r'제\g<1>항'),
        (r'제\s*(\d+)\s+항', r'제\g<1>항'),
        (r'제\s+(\d+)\s*호', r'제\g<1>호'),
        (r'제\s*(\d+)\s+호', r'제\g<1>호'),
        # 금액 표기
        (r'금(\d)', r'금 \g<1>'),
        # 날짜 표기
        (r'(\d{4})\s*[.·]\s*(\d{1,2})\s*[.·]\s*(\d{1,2})', r'\g<1>. \g<2>. \g<3>.'),
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

    def correct_html_formatting(
        self,
        html: str,
        original_image_path: Optional[str] = None,
        original_image_bytes: Optional[bytes] = None
    ) -> str:
        """
        Correct HTML formatting using Gemini's multimodal capabilities.
        Compares the HTML with the original PDF image to ensure visual fidelity.

        Checks and corrects:
        1. Text alignment (center, right, left)
        2. Line spacing / empty lines / multiple spaces
        3. Invisible tables for government forms
        4. Font size and heading detection
        5. Bold text preservation

        Args:
            html: The generated HTML to correct
            original_image_path: Path to original PDF page image (optional)
            original_image_bytes: Raw bytes of original image (optional)

        Returns:
            Corrected HTML string
        """
        if self.model is None:
            logger.warning("Gemini model not available, returning original HTML")
            return html

        try:
            import PIL.Image
            import io

            # Build the prompt content
            content_parts = []

            # Add the original image if provided (for multimodal comparison)
            if original_image_path:
                try:
                    image = PIL.Image.open(original_image_path)
                    content_parts.append(image)
                    content_parts.append("\n위 이미지는 원본 PDF 페이지입니다.\n\n")
                except Exception as e:
                    logger.warning(f"Failed to load image from path: {e}")

            elif original_image_bytes:
                try:
                    image = PIL.Image.open(io.BytesIO(original_image_bytes))
                    content_parts.append(image)
                    content_parts.append("\n위 이미지는 원본 PDF 페이지입니다.\n\n")
                except Exception as e:
                    logger.warning(f"Failed to load image from bytes: {e}")

            # Add the HTML formatting prompt and the HTML to correct
            content_parts.append(self.HTML_FORMATTING_PROMPT)
            content_parts.append(f"\n\n[교정할 HTML]\n{html}")

            # Call Gemini
            for attempt in range(self.max_retries):
                try:
                    if len(content_parts) > 1:
                        # Multimodal request with image
                        response = self.model.generate_content(content_parts)
                    else:
                        # Text-only request
                        response = self.model.generate_content(
                            self.HTML_FORMATTING_PROMPT + f"\n\n[교정할 HTML]\n{html}"
                        )

                    if response.text:
                        corrected_html = response.text.strip()

                        # Remove markdown code blocks if present
                        if corrected_html.startswith("```html"):
                            corrected_html = corrected_html[7:]
                        if corrected_html.startswith("```"):
                            corrected_html = corrected_html[3:]
                        if corrected_html.endswith("```"):
                            corrected_html = corrected_html[:-3]

                        return corrected_html.strip()
                    else:
                        logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")

                except Exception as e:
                    logger.warning(f"Gemini HTML correction error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))

        except ImportError:
            logger.warning("PIL not installed, skipping multimodal HTML correction")

        return html

    def verify_html_formatting(
        self,
        html: str,
        original_image_path: Optional[str] = None
    ) -> Dict:
        """
        Verify HTML formatting against original image.
        Returns a report of formatting issues found.

        Args:
            html: HTML to verify
            original_image_path: Path to original PDF page image

        Returns:
            Dictionary with verification results
        """
        verification_prompt = """다음 HTML의 서식을 분석하고, 아래 5가지 항목에 대해 검증 결과를 JSON 형식으로 출력하시오.

[검증 항목]
1. alignment_issues: 정렬 문제가 있는 요소 목록
2. spacing_issues: 줄 간격/빈 줄 문제 목록
3. table_issues: 선 없는 표 처리 문제 목록
4. font_size_issues: 글씨 크기/제목 처리 문제 목록
5. bold_issues: 볼드체 처리 문제 목록

[HTML]
""" + html + """

[출력 형식 - JSON만 출력]
{
  "alignment_issues": [...],
  "spacing_issues": [...],
  "table_issues": [...],
  "font_size_issues": [...],
  "bold_issues": [...],
  "overall_score": 0-100,
  "needs_correction": true/false
}"""

        if self.model is None:
            return {"error": "Gemini model not available"}

        try:
            response = self.model.generate_content(verification_prompt)
            if response.text:
                import json
                # Try to parse JSON from response
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]

                return json.loads(text.strip())
        except Exception as e:
            logger.error(f"HTML verification error: {e}")

        return {"error": "Verification failed"}


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


# Convenience function for HTML formatting correction
def correct_html_formatting(
    html: str,
    original_image_path: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Convenience function to correct HTML formatting using Gemini multimodal.

    Verifies and corrects:
    1. Text alignment (center, right, left)
    2. Line spacing / empty lines / multiple spaces
    3. Invisible tables for government forms
    4. Font size and heading detection
    5. Bold text preservation

    Args:
        html: HTML to correct
        original_image_path: Path to original PDF page image for comparison
        api_key: Gemini API key (optional)

    Returns:
        Corrected HTML string
    """
    corrector = GeminiCorrector(api_key=api_key)
    return corrector.correct_html_formatting(html, original_image_path=original_image_path)
