# 법률 문서 OCR 서비스

**초가성비 법률 문서 디지털 변환기** - 99.99% 정확도를 목표로 하는 프로덕션 레벨 OCR 서비스

## 주요 기능

- **대용량 PDF 처리**: 5,000+ 페이지 PDF 지원
- **하이브리드 OCR**: GOT-OCR (표) + PaddleOCR (한글 텍스트) 이원화 전략
- **스마트 레이아웃 분석**: YOLOv8 + Heuristic 기반 테이블 탐지 (선 없는 표 포함)
- **AI 교정**: Gemini 2.0 Flash를 활용한 법률 문서 특화 오타 교정
- **실시간 진행률**: 페이지별 처리 진행 상황 실시간 확인
- **비동기 처리**: Celery + Redis 기반 작업 큐 시스템
- **저렴한 비용**: 페이지당 50원

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                   FastAPI Server (main.py)                   │
│  - 파일 업로드 (PDF, 이미지)                                  │
│  - 작업 상태 조회                                             │
│  - 결과 다운로드                                              │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     Redis (Message Broker)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                  Celery Worker (tasks.py)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            PDF Processor (pdf2image/PyMuPDF)         │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │                                 │
│  ┌─────────────────────────▼───────────────────────────┐    │
│  │            Layout Analyzer (YOLOv8)                  │    │
│  │  - 표 영역 탐지 (bordered)                           │    │
│  │  - 선 없는 표 탐지 (heuristic)                       │    │
│  └────────┬────────────────────────┬───────────────────┘    │
│           │                        │                         │
│  ┌────────▼────────┐      ┌───────▼───────────┐            │
│  │   GOT-OCR       │      │    PaddleOCR       │            │
│  │  (표 인식)       │      │   (텍스트 인식)     │            │
│  └────────┬────────┘      └───────┬───────────┘            │
│           │                        │                         │
│  ┌────────▼────────────────────────▼───────────────────┐    │
│  │              Legal Text Parser                       │    │
│  │  - 제목/본문 구분 (h1, h2, p)                        │    │
│  │  - 볼드체 인식                                       │    │
│  │  - 들여쓰기 감지                                     │    │
│  │  - 법률 오타 교정 (甲乙, 로마숫자 등)                 │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │                                 │
│  ┌─────────────────────────▼───────────────────────────┐    │
│  │           Gemini 2.0 Flash (최종 교정)               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 빠른 시작

### 1. 사전 요구사항

- Docker & Docker Compose
- NVIDIA GPU + NVIDIA Docker (권장)
- Gemini API Key (선택)

### 2. 모델 다운로드

```bash
# GOT-OCR 모델 다운로드 (HuggingFace)
# https://huggingface.co/stepfun-ai/GOT-OCR2_0 에서 다운로드
# weights/GOT-OCR2_0/ 폴더에 모든 파일 배치

# YOLO 모델 (기본 제공 또는 커스텀)
# weights/yolo_table_best.pt/ 폴더 확인
```

### 3. 환경 설정

```bash
# 환경 변수 설정
cp .env.example .env

# .env 파일 편집
nano .env
# GEMINI_API_KEY=your_api_key_here
```

### 4. Docker 실행

```bash
# 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d
```

### 5. 서비스 접속

- **웹 UI**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **작업 모니터링**: http://localhost:5555 (Flower)

## API 사용법

### 파일 업로드 (PDF 또는 이미지)

```bash
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@document.pdf" \
  -F "apply_gemini=true"
```

응답:
```json
{
  "task_id": "abc123...",
  "status": "processing",
  "message": "PDF 처리가 시작되었습니다. 총 100페이지",
  "estimated_pages": 100,
  "estimated_price_krw": 5000
}
```

### 작업 상태 확인

```bash
curl "http://localhost:8000/ocr/status/{task_id}"
```

응답:
```json
{
  "task_id": "abc123...",
  "state": "processing",
  "progress": 45,
  "current_page": 45,
  "total_pages": 100,
  "status": "페이지 45 처리 중..."
}
```

### 결과 조회

```bash
# JSON 형식
curl "http://localhost:8000/ocr/result/{task_id}?format=json"

# 마크다운만
curl "http://localhost:8000/ocr/result/{task_id}?format=markdown"

# HTML 렌더링
curl "http://localhost:8000/ocr/result/{task_id}?format=html"
```

### 여러 이미지 배치 처리

```bash
curl -X POST "http://localhost:8000/ocr/upload-multiple" \
  -F "files=@page1.png" \
  -F "files=@page2.png" \
  -F "files=@page3.png"
```

## 로컬 개발 (Docker 없이)

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Redis 실행 (별도 터미널)
redis-server

# Celery Worker 실행 (별도 터미널)
celery -A tasks worker --loglevel=info --concurrency=1

# API 서버 실행
python main.py
```

## 프로젝트 구조

```
OCR/
├── main.py              # FastAPI 서버
├── tasks.py             # Celery 비동기 작업
├── ocr_engine.py        # OCR 파이프라인 (GOT-OCR + PaddleOCR)
├── layout_analyzer.py   # 레이아웃 분석 (YOLOv8 + Heuristic)
├── pdf_processor.py     # PDF 처리 (pdf2image/PyMuPDF)
├── config.py            # 설정 관리
├── requirements.txt     # Python 의존성
├── Dockerfile           # Docker 이미지 설정
├── docker-compose.yml   # Docker Compose 설정
├── .env.example         # 환경 변수 예시
├── templates/
│   └── index.html       # 프론트엔드 UI
├── weights/
│   ├── GOT-OCR2_0/      # GOT-OCR 모델 파일
│   └── yolo_table_best.pt/
│       └── yolov8n.pt   # YOLO 테이블 탐지 모델
└── README.md
```

## 가격 정책

| 페이지 수 | 단가 | 할인 |
|-----------|------|------|
| 1-99      | 50원 | - |
| 100-499   | 45원 | 10% |
| 500-999   | 40원 | 20% |
| 1000+     | 35원 | 30% |

## 기술 스택

- **OCR 엔진**
  - [GOT-OCR2.0](https://huggingface.co/stepfun-ai/GOT-OCR2_0) - 표/수식 인식
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 한글 텍스트 인식

- **레이아웃 분석**
  - [YOLOv8](https://ultralytics.com/) - 테이블 영역 탐지
  - OpenCV - 선 없는 표 휴리스틱 탐지

- **LLM 교정**
  - Gemini 2.0 Flash - 법률 문서 특화 오타 교정

- **백엔드**
  - FastAPI - REST API 서버
  - Celery - 비동기 작업 큐
  - Redis - 메시지 브로커

- **인프라**
  - Docker - 컨테이너화
  - NVIDIA Docker - GPU 지원

## 성능 최적화 팁

1. **GPU 메모리**: GOT-OCR + PaddleOCR은 약 8GB VRAM 필요
2. **배치 크기**: `MAX_PAGES_PER_BATCH=50`으로 메모리 관리
3. **워커 수**: GPU당 1개 워커 권장 (`--concurrency=1`)
4. **PDF DPI**: 300 DPI 권장 (OCR 정확도와 속도 균형)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

이슈 및 PR 환영합니다!
