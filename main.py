from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tasks import process_ocr_task, celery_app
from celery.result import AsyncResult
import shutil
import os
import uuid

app = FastAPI(title="Legal OCR API")

UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ocr/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """이미지를 업로드받아 OCR 작업 큐에 등록"""
    try:
        # 고유 파일명 생성
        ext = file.filename.split('.')[-1]
        task_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{task_id}.{ext}")
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Celery Task 실행 (비동기)
        # file_path를 넘겨주어 워커가 처리하게 함
        task = process_ocr_task.delay(file_path)
        
        return {"task_id": task.id, "status": "processing"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ocr/status/{task_id}")
async def get_status(task_id: str):
    """작업 상태 및 결과 조회"""
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.state == 'PENDING':
        return {"status": "pending", "progress": 0}
    elif task_result.state == 'SUCCESS':
        return {"status": "completed", "result": task_result.result}
    elif task_result.state == 'FAILURE':
        return {"status": "failed", "error": str(task_result.info)}
    
    return {"status": task_result.state}