from celery import Celery
import os
import google.generativeai as genai
from ocr_engine import HybridOCRPipeline
from dotenv import load_dotenv

load_dotenv()

# Redis 연결 설정
celery_app = Celery(
    "ocr_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

# 모델 경로 설정 (서버 환경에 맞게 수정)
GOT_MODEL_PATH = "./weights/GOT-OCR2_0"
YOLO_MODEL_PATH = "./weights/yolo_table_best.pt"

# 전역 변수로 파이프라인 로드 (워커 시작 시 1회만 로딩)
# 주의: GPU 메모리를 많이 먹으므로 워커 프로세스 개수 조절 필요
pipeline = None

@celery_app.task(bind=True)
def process_ocr_task(self, image_path):
    global pipeline
    if pipeline is None:
        print("Initializing OCR Pipeline in Worker...")
        pipeline = HybridOCRPipeline(GOT_MODEL_PATH, YOLO_MODEL_PATH)

    # 1. OCR 수행 (Hybrid)
    raw_markdown = pipeline.run(image_path)
    
    # 2. Gemini 2.0 Flash 교정
    final_result = correct_with_gemini(raw_markdown)
    
    # 3. 임시 파일 정리
    if os.path.exists(image_path):
        os.remove(image_path)
        
    return final_result

def correct_with_gemini(text):
    """Gemini 2.0 Flash API 호출"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return text # 키 없으면 원본 반환

    genai.configure(api_key=api_key)
    
    # 2.0 Flash 모델 설정 (정식 명칭 확인 필요, 현재 exp or flash)
    model = genai.GenerativeModel('gemini-2.0-flash-exp') 

    system_prompt = """
    당신은 법률 문서 교정 전문가입니다. 아래 OCR된 마크다운 텍스트를 교정하십시오.
    [규칙]
    1. 갑(甲), 을(乙) 등 인물/회사 명칭의 오타(Z, E 등)를 한자로 복원.
    2. 로마자(I, II)를 유니코드(Ⅰ, Ⅱ)로 복원.
    3. 법조문 '제1o조' -> '제10조' 등 숫자 오타 수정.
    4. 마크다운 표(Table) 구조가 깨졌다면 문맥에 맞게 수정.
    5. 내용은 절대 요약하지 말고 원문 유지.
    """
    
    try:
        response = model.generate_content(f"{system_prompt}\n\n[TEXT_START]\n{text}\n[TEXT_END]")
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return text