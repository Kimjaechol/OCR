"""
Legal Document OCR - Celery Tasks Module
=========================================
Async task processing with progress tracking for large documents
Supports 5000+ page PDF processing with Gemini LLM correction
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
from loguru import logger

from celery import Celery, states
from celery.exceptions import Ignore
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("google-generativeai not installed")
    GEMINI_AVAILABLE = False

from config import get_settings

# Load environment variables
load_dotenv()

# Get settings
settings = get_settings()

# ============================================
# Celery Configuration
# ============================================
celery_app = Celery(
    "legal_ocr_tasks",
    broker=settings.redis_url,
    backend=settings.redis_url
)

# Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Seoul',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # Process one task at a time (GPU)
    result_expires=86400,  # Results expire after 24 hours
)

# ============================================
# Global Pipeline Instance (Lazy Loading)
# ============================================
_pipeline = None
_pdf_processor = None


def get_pipeline():
    """Get or create OCR pipeline instance (singleton per worker)"""
    global _pipeline
    if _pipeline is None:
        from ocr_engine import HybridOCRPipeline
        logger.info("Initializing OCR Pipeline in worker...")
        _pipeline = HybridOCRPipeline(
            got_model_path=settings.got_model_path,
            yolo_model_path=settings.yolo_model_path,
            use_gpu=True
        )
        _pipeline.initialize()
    return _pipeline


def get_pdf_processor():
    """Get or create PDF processor instance"""
    global _pdf_processor
    if _pdf_processor is None:
        from pdf_processor import PDFProcessor
        _pdf_processor = PDFProcessor(
            dpi=settings.ocr_dpi,
            batch_size=settings.max_pages_per_batch
        )
    return _pdf_processor


# ============================================
# Gemini LLM Correction
# ============================================
def correct_with_gemini(text: str) -> str:
    """
    Correct OCR text using Gemini 2.0 Flash.

    Args:
        text: Raw OCR markdown text

    Returns:
        Corrected text
    """
    if not settings.enable_gemini_correction:
        return text

    api_key = settings.gemini_api_key
    if not api_key or not GEMINI_AVAILABLE:
        logger.warning("Gemini correction unavailable (no API key or library)")
        return text

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        system_prompt = """당신은 법률 문서 교정 전문가입니다. 아래 OCR된 마크다운 텍스트를 교정하십시오.

[규칙]
1. 갑(甲), 을(乙) 등 인물/회사 명칭의 오타(Z, E 등)를 한자로 복원.
2. 로마자(I, II)를 유니코드(Ⅰ, Ⅱ)로 복원.
3. 법조문 '제1o조' -> '제10조' 등 숫자 오타 수정.
4. 마크다운 표(Table) 구조가 깨졌다면 문맥에 맞게 수정.
5. 내용은 절대 요약하지 말고 원문 유지.
6. 교정된 텍스트만 출력하고 설명은 하지 마세요."""

        response = model.generate_content(
            f"{system_prompt}\n\n[원문 시작]\n{text}\n[원문 끝]",
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 8192,
            }
        )

        return response.text

    except Exception as e:
        logger.error(f"Gemini correction error: {e}")
        return text


# ============================================
# Celery Tasks
# ============================================
@celery_app.task(bind=True, name='tasks.process_single_image')
def process_single_image_task(
    self,
    image_path: str,
    apply_gemini: bool = True
) -> Dict[str, Any]:
    """
    Process a single image file.

    Args:
        image_path: Path to image file
        apply_gemini: Whether to apply Gemini correction

    Returns:
        Dict with OCR results
    """
    try:
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'status': 'OCR 처리 중...'}
        )

        pipeline = get_pipeline()
        result = pipeline.process_image(image_path, page_number=1)

        self.update_state(
            state='PROCESSING',
            meta={'progress': 80, 'status': 'LLM 교정 중...'}
        )

        # Apply Gemini correction
        final_markdown = result.markdown
        if apply_gemini:
            final_markdown = correct_with_gemini(result.markdown)

        # Cleanup temp file
        if os.path.exists(image_path):
            os.remove(image_path)

        return {
            'status': 'completed',
            'markdown': final_markdown,
            'raw_text': result.raw_text,
            'tables_count': result.tables_count,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'page_count': 1
        }

    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


@celery_app.task(bind=True, name='tasks.process_pdf_document')
def process_pdf_document_task(
    self,
    pdf_path: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    apply_gemini: bool = True
) -> Dict[str, Any]:
    """
    Process a PDF document with progress tracking.

    Args:
        pdf_path: Path to PDF file
        start_page: First page to process
        end_page: Last page to process (None = all)
        apply_gemini: Whether to apply Gemini correction

    Returns:
        Dict with OCR results for all pages
    """
    start_time = time.time()

    try:
        # Initialize processors
        pipeline = get_pipeline()
        pdf_processor = get_pdf_processor()

        # Get PDF info
        pdf_info = pdf_processor.get_pdf_info(pdf_path)
        total_pages = pdf_info.page_count

        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(end_page, total_pages)

        pages_to_process = end_page - start_page + 1

        logger.info(
            f"Starting PDF processing: {pdf_path}, "
            f"pages {start_page}-{end_page} of {total_pages}"
        )

        # Update initial state
        self.update_state(
            state='PROCESSING',
            meta={
                'progress': 0,
                'current_page': 0,
                'total_pages': pages_to_process,
                'status': 'PDF 분석 중...'
            }
        )

        # Process pages
        all_results = []
        combined_markdown = []

        for page_image in pdf_processor.iter_pages(pdf_path, start_page, end_page):
            page_num = page_image.page_number
            progress = int(((page_num - start_page) / pages_to_process) * 90)

            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': progress,
                    'current_page': page_num - start_page + 1,
                    'total_pages': pages_to_process,
                    'status': f'페이지 {page_num} 처리 중...'
                }
            )

            try:
                result = pipeline.process_image(
                    page_image.image,
                    page_number=page_num
                )

                all_results.append({
                    'page': page_num,
                    'markdown': result.markdown,
                    'tables_count': result.tables_count,
                    'confidence': result.confidence
                })

                # Add page header to combined markdown
                combined_markdown.append(
                    f"\n\n---\n\n## 페이지 {page_num}\n\n{result.markdown}"
                )

            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                all_results.append({
                    'page': page_num,
                    'markdown': f'[처리 실패: {str(e)}]',
                    'error': str(e)
                })
                combined_markdown.append(
                    f"\n\n---\n\n## 페이지 {page_num}\n\n[처리 실패: {str(e)}]"
                )

        # Combine all markdown
        full_markdown = "\n".join(combined_markdown)

        # Apply Gemini correction to combined result
        if apply_gemini and len(full_markdown) < 30000:  # Limit for API
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': 95,
                    'current_page': pages_to_process,
                    'total_pages': pages_to_process,
                    'status': 'LLM 교정 중...'
                }
            )
            full_markdown = correct_with_gemini(full_markdown)

        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        processing_time = time.time() - start_time

        # Calculate stats
        total_tables = sum(r.get('tables_count', 0) for r in all_results)
        avg_confidence = sum(
            r.get('confidence', 0) for r in all_results
        ) / len(all_results) if all_results else 0

        return {
            'status': 'completed',
            'markdown': full_markdown,
            'pages': all_results,
            'page_count': pages_to_process,
            'total_tables': total_tables,
            'average_confidence': avg_confidence,
            'processing_time': processing_time,
            'price_krw': pages_to_process * settings.price_per_page
        }

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


@celery_app.task(bind=True, name='tasks.process_batch_images')
def process_batch_images_task(
    self,
    image_paths: List[str],
    apply_gemini: bool = True
) -> Dict[str, Any]:
    """
    Process multiple image files.

    Args:
        image_paths: List of image file paths
        apply_gemini: Whether to apply Gemini correction

    Returns:
        Dict with combined OCR results
    """
    start_time = time.time()
    total_images = len(image_paths)

    try:
        pipeline = get_pipeline()
        all_results = []
        combined_markdown = []

        for i, image_path in enumerate(image_paths):
            progress = int((i / total_images) * 90)

            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': progress,
                    'current_page': i + 1,
                    'total_pages': total_images,
                    'status': f'이미지 {i + 1}/{total_images} 처리 중...'
                }
            )

            try:
                result = pipeline.process_image(image_path, page_number=i + 1)

                all_results.append({
                    'page': i + 1,
                    'markdown': result.markdown,
                    'tables_count': result.tables_count,
                    'confidence': result.confidence
                })

                combined_markdown.append(
                    f"\n\n---\n\n## 페이지 {i + 1}\n\n{result.markdown}"
                )

                # Cleanup processed file
                if os.path.exists(image_path):
                    os.remove(image_path)

            except Exception as e:
                logger.error(f"Failed to process image {i + 1}: {e}")
                all_results.append({
                    'page': i + 1,
                    'error': str(e)
                })

        # Combine markdown
        full_markdown = "\n".join(combined_markdown)

        # Apply Gemini correction
        if apply_gemini and len(full_markdown) < 30000:
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': 95,
                    'current_page': total_images,
                    'total_pages': total_images,
                    'status': 'LLM 교정 중...'
                }
            )
            full_markdown = correct_with_gemini(full_markdown)

        processing_time = time.time() - start_time

        return {
            'status': 'completed',
            'markdown': full_markdown,
            'pages': all_results,
            'page_count': total_images,
            'processing_time': processing_time,
            'price_krw': total_images * settings.price_per_page
        }

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


# ============================================
# Task Status Helper
# ============================================
def get_task_progress(task_id: str) -> Dict[str, Any]:
    """
    Get detailed progress for a task.

    Args:
        task_id: Celery task ID

    Returns:
        Dict with progress information
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    if result.state == 'PENDING':
        return {
            'state': 'pending',
            'progress': 0,
            'status': '대기 중...'
        }
    elif result.state == 'PROCESSING':
        meta = result.info or {}
        return {
            'state': 'processing',
            'progress': meta.get('progress', 0),
            'current_page': meta.get('current_page', 0),
            'total_pages': meta.get('total_pages', 0),
            'status': meta.get('status', '처리 중...')
        }
    elif result.state == 'SUCCESS':
        return {
            'state': 'completed',
            'progress': 100,
            'status': '완료',
            'result': result.result
        }
    elif result.state == 'FAILURE':
        return {
            'state': 'failed',
            'progress': 0,
            'status': '실패',
            'error': str(result.info)
        }
    else:
        return {
            'state': result.state.lower(),
            'progress': 0,
            'status': result.state
        }
