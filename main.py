"""
Legal Document OCR - FastAPI Server
====================================
Production-ready API for legal document OCR processing
Supports PDF (5000+ pages), images, and batch processing
"""

import os
import uuid
import shutil
import aiofiles
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from config import get_settings
from tasks import (
    celery_app,
    process_single_image_task,
    process_pdf_document_task,
    process_batch_images_task,
    process_folder_task,
    get_task_progress
)

# Get settings
settings = get_settings()


# ============================================
# Lifespan Context Manager
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown"""
    # Startup
    logger.info("Legal Document OCR API starting up...")
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Output directory: {settings.output_dir}")
    yield
    # Shutdown
    logger.info("Legal Document OCR API shutting down...")


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="Legal Document OCR API",
    description="초가성비 법률 문서 OCR 서비스 - 99.99% 정확도 목표",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================
# Pydantic Models
# ============================================
class TaskResponse(BaseModel):
    """Response model for task creation"""
    task_id: str
    status: str
    message: str
    estimated_pages: Optional[int] = None
    estimated_price_krw: Optional[int] = None


class ProgressResponse(BaseModel):
    """Response model for task progress"""
    task_id: str
    state: str
    progress: int
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str


# ============================================
# Utility Functions
# ============================================
def get_file_extension(filename: str) -> str:
    """Extract file extension"""
    return Path(filename).suffix.lower()


def is_pdf(filename: str) -> bool:
    """Check if file is PDF"""
    return get_file_extension(filename) == '.pdf'


def is_image(filename: str) -> bool:
    """Check if file is an image"""
    return get_file_extension(filename) in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file asynchronously"""
    async with aiofiles.open(destination, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    return destination


# ============================================
# API Endpoints
# ============================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    html_path = Path("templates/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Legal Document OCR</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>Legal Document OCR API</h1>
        <p>API documentation: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/ocr/upload", response_model=TaskResponse)
@app.post("/ocr/analyze", response_model=TaskResponse)  # Alias for Electron client
async def upload_document(
    file: UploadFile = File(...),
    start_page: int = Query(1, ge=1, description="시작 페이지 (PDF만)"),
    end_page: Optional[int] = Query(None, ge=1, description="종료 페이지 (PDF만)"),
    apply_gemini: bool = Query(True, description="Gemini LLM 교정 적용")
):
    """
    문서 업로드 및 OCR 처리 시작

    - PDF 파일: 최대 5000+ 페이지 지원
    - 이미지 파일: PNG, JPG, JPEG, TIFF 등 지원
    - 처리 비용: 페이지당 50원
    - /ocr/analyze는 Electron 데스크탑 클라이언트 호환용 alias
    """
    try:
        # Validate file type
        filename = file.filename or "unknown"
        if not (is_pdf(filename) or is_image(filename)):
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 파일 형식입니다. PDF 또는 이미지 파일을 업로드하세요."
            )

        # Generate unique file path
        task_id = str(uuid.uuid4())
        ext = get_file_extension(filename)
        file_path = os.path.join(settings.upload_dir, f"{task_id}{ext}")

        # Save file
        await save_upload_file(file, file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        logger.info(f"File uploaded: {filename}, size: {file_size_mb:.2f}MB, task_id: {task_id}")

        # Process based on file type
        if is_pdf(filename):
            # Get PDF info for estimation
            from pdf_processor import PDFProcessor
            processor = PDFProcessor()
            try:
                pdf_info = processor.get_pdf_info(file_path)
                total_pages = pdf_info.page_count

                if end_page is None:
                    end_page = total_pages
                else:
                    end_page = min(end_page, total_pages)

                pages_to_process = end_page - start_page + 1
                estimated_price = pages_to_process * settings.price_per_page

            except Exception as e:
                logger.error(f"PDF info error: {e}")
                pages_to_process = None
                estimated_price = None

            # Queue PDF processing task
            task = process_pdf_document_task.delay(
                file_path,
                start_page=start_page,
                end_page=end_page,
                apply_gemini=apply_gemini
            )

            return TaskResponse(
                task_id=task.id,
                status="processing",
                message=f"PDF 처리가 시작되었습니다. 총 {pages_to_process or '?'}페이지",
                estimated_pages=pages_to_process,
                estimated_price_krw=estimated_price
            )

        else:
            # Queue image processing task
            task = process_single_image_task.delay(
                file_path,
                apply_gemini=apply_gemini
            )

            return TaskResponse(
                task_id=task.id,
                status="processing",
                message="이미지 처리가 시작되었습니다.",
                estimated_pages=1,
                estimated_price_krw=settings.price_per_page
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/upload-multiple", response_model=TaskResponse)
async def upload_multiple_images(
    files: List[UploadFile] = File(...),
    apply_gemini: bool = Query(True, description="Gemini LLM 교정 적용")
):
    """
    여러 이미지 파일 업로드 및 배치 OCR 처리

    - 최대 100개 이미지 동시 업로드 가능
    - 처리 비용: 이미지당 50원
    """
    try:
        if len(files) > 100:
            raise HTTPException(
                status_code=400,
                detail="한 번에 최대 100개 파일까지 업로드 가능합니다."
            )

        task_id = str(uuid.uuid4())
        saved_paths = []

        for i, file in enumerate(files):
            filename = file.filename or f"image_{i}"

            if not is_image(filename):
                continue

            ext = get_file_extension(filename)
            file_path = os.path.join(settings.upload_dir, f"{task_id}_{i:04d}{ext}")
            await save_upload_file(file, file_path)
            saved_paths.append(file_path)

        if not saved_paths:
            raise HTTPException(
                status_code=400,
                detail="유효한 이미지 파일이 없습니다."
            )

        # Queue batch processing task
        task = process_batch_images_task.delay(
            saved_paths,
            apply_gemini=apply_gemini
        )

        return TaskResponse(
            task_id=task.id,
            status="processing",
            message=f"{len(saved_paths)}개 이미지 처리가 시작되었습니다.",
            estimated_pages=len(saved_paths),
            estimated_price_krw=len(saved_paths) * settings.price_per_page
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multiple upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class FolderProcessRequest(BaseModel):
    """Request model for folder processing"""
    folder_path: str
    recursive: bool = True
    apply_gemini: bool = True


@app.post("/ocr/process-folder", response_model=TaskResponse)
async def process_folder(request: FolderProcessRequest):
    """
    서버 내 폴더의 모든 PDF/이미지 파일 OCR 처리

    - folder_path: 처리할 폴더 경로
    - recursive: 하위 폴더 포함 여부 (기본: True)
    - 지원 형식: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP
    """
    try:
        folder_path = request.folder_path

        if not os.path.exists(folder_path):
            raise HTTPException(
                status_code=400,
                detail=f"폴더를 찾을 수 없습니다: {folder_path}"
            )

        if not os.path.isdir(folder_path):
            raise HTTPException(
                status_code=400,
                detail=f"유효한 폴더 경로가 아닙니다: {folder_path}"
            )

        # Scan folder for estimation
        from batch_processor import BatchProcessor
        processor = BatchProcessor()
        scan_result = processor.scan_folder(folder_path, request.recursive)

        if scan_result.total_files == 0:
            raise HTTPException(
                status_code=400,
                detail="처리할 파일이 없습니다. (PDF, 이미지 파일 없음)"
            )

        logger.info(
            f"Folder processing requested: {folder_path}, "
            f"{scan_result.total_files} files, ~{scan_result.estimated_pages} pages"
        )

        # Queue folder processing task
        task = process_folder_task.delay(
            folder_path,
            recursive=request.recursive,
            apply_gemini=request.apply_gemini
        )

        estimated_price = scan_result.estimated_pages * settings.price_per_page

        return TaskResponse(
            task_id=task.id,
            status="processing",
            message=f"폴더 처리가 시작되었습니다. "
                    f"PDF {len(scan_result.pdf_files)}개, "
                    f"이미지 {len(scan_result.image_files)}개, "
                    f"예상 {scan_result.estimated_pages}페이지",
            estimated_pages=scan_result.estimated_pages,
            estimated_price_krw=estimated_price
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Folder processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ocr/scan-folder")
async def scan_folder(
    folder_path: str = Query(..., description="스캔할 폴더 경로"),
    recursive: bool = Query(True, description="하위 폴더 포함 여부")
):
    """
    폴더 내 처리 가능한 파일 목록 조회 (처리 전 확인용)

    - 처리 예상 시간 및 비용 확인
    - PDF/이미지 파일 수 확인
    """
    try:
        if not os.path.exists(folder_path):
            raise HTTPException(
                status_code=400,
                detail=f"폴더를 찾을 수 없습니다: {folder_path}"
            )

        from batch_processor import BatchProcessor
        processor = BatchProcessor()
        scan_result = processor.scan_folder(folder_path, recursive)

        return {
            "folder_path": scan_result.folder_path,
            "pdf_files": len(scan_result.pdf_files),
            "image_files": len(scan_result.image_files),
            "total_files": scan_result.total_files,
            "total_size_mb": round(scan_result.total_size_mb, 2),
            "estimated_pages": scan_result.estimated_pages,
            "estimated_price_krw": scan_result.estimated_pages * settings.price_per_page,
            "file_list": {
                "pdfs": scan_result.pdf_files[:20],  # First 20 only
                "images": scan_result.image_files[:20]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Folder scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ocr/status/{task_id}")
async def get_status(task_id: str):
    """
    작업 상태 및 진행률 조회

    - progress: 0-100 진행률
    - state: pending, processing, completed, failed
    - status: state와 동일 (Electron 클라이언트 호환용)
    - result: 완료 시 마크다운 텍스트 또는 전체 결과
    """
    try:
        progress_info = get_task_progress(task_id)
        state = progress_info.get('state', 'unknown')
        result_data = progress_info.get('result', {})

        # Extract markdown for simple result (Electron client compatibility)
        simple_result = None
        if state == 'completed' and result_data:
            simple_result = result_data.get('markdown', '')

        return {
            "task_id": task_id,
            "state": state,
            "status": state,  # Electron 클라이언트 호환 (status == state)
            "progress": progress_info.get('progress', 0),
            "current_page": progress_info.get('current_page'),
            "total_pages": progress_info.get('total_pages'),
            "status_message": progress_info.get('status', ''),
            "result": simple_result,  # 마크다운 텍스트 직접 반환
            "full_result": result_data,  # 전체 결과 (page별 상세 정보)
            "error": progress_info.get('error')
        }

    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ocr/result/{task_id}")
async def get_result(
    task_id: str,
    format: str = Query("json", description="출력 형식: json, markdown, html")
):
    """
    완료된 작업의 결과 조회

    - format=json: 전체 JSON 결과
    - format=markdown: 마크다운 텍스트만
    - format=html: HTML 렌더링
    """
    try:
        progress_info = get_task_progress(task_id)

        if progress_info.get('state') != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"작업이 아직 완료되지 않았습니다. 현재 상태: {progress_info.get('state')}"
            )

        result = progress_info.get('result', {})

        if format == "markdown":
            return JSONResponse(content={
                "markdown": result.get('markdown', ''),
                "page_count": result.get('page_count', 0)
            })

        elif format == "html":
            # Return stored HTML (with proper formatting preserved)
            html_content = result.get('html', '')
            if not html_content:
                # Fallback: simple conversion if HTML not available
                markdown_text = result.get('markdown', '')
                html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OCR 결과</title></head>
<body><pre>{markdown_text}</pre></body></html>"""
            return HTMLResponse(content=html_content)

        else:
            return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Result fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ocr/download/{task_id}")
async def download_result(
    task_id: str,
    format: str = Query("markdown", description="다운로드 형식: markdown, html, json")
):
    """
    완료된 작업 결과를 파일로 다운로드

    - format=markdown: .md 파일 다운로드
    - format=html: .html 파일 다운로드
    - format=json: .json 파일 다운로드
    """
    try:
        progress_info = get_task_progress(task_id)

        if progress_info.get('state') != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"작업이 아직 완료되지 않았습니다. 현재 상태: {progress_info.get('state')}"
            )

        result = progress_info.get('result', {})

        if format == "markdown":
            content = result.get('markdown', '')
            filename = f"ocr_result_{task_id[:8]}.md"
            media_type = "text/markdown"
        elif format == "html":
            content = result.get('html', '')
            filename = f"ocr_result_{task_id[:8]}.html"
            media_type = "text/html"
        else:
            import json as json_module
            content = json_module.dumps(result, ensure_ascii=False, indent=2)
            filename = f"ocr_result_{task_id[:8]}.json"
            media_type = "application/json"

        # Create temp file for download
        temp_path = os.path.join(settings.output_dir, filename)
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return FileResponse(
            path=temp_path,
            filename=filename,
            media_type=media_type
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ocr/cancel/{task_id}")
async def cancel_task(task_id: str):
    """작업 취소"""
    try:
        from celery.result import AsyncResult
        result = AsyncResult(task_id, app=celery_app)
        result.revoke(terminate=True)

        return {"status": "cancelled", "task_id": task_id}

    except Exception as e:
        logger.error(f"Cancel error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ocr/pricing")
async def get_pricing():
    """가격 정보 조회"""
    return {
        "price_per_page_krw": settings.price_per_page,
        "currency": "KRW",
        "description": "페이지당 50원 (표, 이미지 포함)",
        "bulk_discount": {
            "100_pages": "10% 할인",
            "500_pages": "20% 할인",
            "1000_pages": "30% 할인"
        }
    }


@app.get("/api/stats")
async def get_stats():
    """시스템 통계"""
    try:
        # Get Celery stats
        inspector = celery_app.control.inspect()
        active = inspector.active() or {}
        reserved = inspector.reserved() or {}

        active_count = sum(len(tasks) for tasks in active.values())
        queued_count = sum(len(tasks) for tasks in reserved.values())

        return {
            "active_tasks": active_count,
            "queued_tasks": queued_count,
            "workers": list(active.keys()) if active else []
        }
    except Exception as e:
        return {
            "active_tasks": 0,
            "queued_tasks": 0,
            "workers": [],
            "error": str(e)
        }


# ============================================
# Error Handlers
# ============================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# ============================================
# Run with Uvicorn
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
