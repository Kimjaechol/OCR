import cv2
import numpy as np
import re
import statistics
import os
import torch
from paddleocr import PaddleOCR
from transformers import AutoModelForCausalLM, AutoTokenizer

# 같은 폴더에 있는 layout_analyzer.py에서 클래스 임포트
try:
    from layout_analyzer import LayoutAnalyzer
except ImportError:
    print("[Warning] layout_analyzer.py를 찾을 수 없습니다. 같은 폴더에 위치시켜주세요.")
    LayoutAnalyzer = None

# =========================================================
# 1. UltimateLegalParser: 텍스트 구조 분석 및 교정 클래스
# =========================================================
class UltimateLegalParser:
    def __init__(self, image):
        self.image = image # OpenCV Grayscale Image
        # 법률 문서 스타일 기준값
        self.H1_SCALE = 1.4       # 본문보다 1.4배 크면 대제목
        self.H2_SCALE = 1.15      # 본문보다 1.15배 크면 소제목
        self.INDENT_PX = 20       # 기준선보다 20px 밀리면 들여쓰기
        self.BOLD_RATIO = 1.10    # 평균 진하기보다 10% 진하면 볼드체

    def get_geometry(self, box):
        """PaddleOCR 박스 좌표에서 기하학적 정보 추출"""
        pts = np.array(box, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        return h, x, y, w, pts

    def is_bold(self, pts, median_density):
        """[OpenCV] 글자 영역의 픽셀 밀도(Density)를 계산하여 볼드체 여부 판별"""
        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, [pts], 255)
        
        # 원본에서 글자만 추출 (마스킹)
        masked_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        
        # 이진화 (글자=흰색, 배경=검정으로 변환)
        _, binary = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        x, y, w, h = cv2.boundingRect(pts)
        area = w * h
        if area == 0: return False
        
        roi = binary[y:y+h, x:x+w]
        density = cv2.countNonZero(roi) / area
        
        # 평균보다 일정 비율 이상 진하면 볼드체
        return density > (median_density * self.BOLD_RATIO)

    def fix_typos(self, text):
        """[Regex] 법률 문서 특화 오타 교정 로직"""
        # 1. 인물/회사 명칭 오타: 갑(甲) 뒤의 Z, E, H 등을 을(乙)로 복원
        if re.search(r'[갑甲]', text): 
            text = re.sub(r'\bZ\b', '乙', text)
            text = re.sub(r'\bE\b', '乙', text)
        
        # 2. 법조문 숫자 오타: '제1o조' -> '제10조'
        text = re.sub(r'(제\s*\d+)[oO](조)', r'\1\2', text)
        
        # 3. 로마자 숫자 복원: I, II -> Ⅰ, Ⅱ
        mapping = {'I': 'Ⅰ', 'II': 'Ⅱ', 'III': 'Ⅲ', 'IV': 'Ⅳ', 'V': 'Ⅴ'}
        words = text.split()
        new_words = []
        for w in words:
            if w in mapping:
                new_words.append(mapping[w])
            else:
                new_words.append(w)
        return " ".join(new_words)

    def parse_text_segments(self, ocr_result):
        """PaddleOCR 결과를 분석하여 구조화된 세그먼트 리스트 반환"""
        if not ocr_result: return []
        
        # 1. 문서 전체 통계 분석
        heights, densities, x_positions = [], [], []
        
        for line in ocr_result:
            box = line[0]
            h, x, y, w, pts = self.get_geometry(box)
            heights.append(h)
            x_positions.append(x)
            
            # 밀도 샘플링
            mask = np.zeros_like(self.image)
            cv2.fillPoly(mask, [pts], 255)
            masked = cv2.bitwise_and(self.image, self.image, mask=mask)
            _, binary = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            if w*h > 0:
                d = cv2.countNonZero(binary[y:y+h, x:x+w]) / (w*h)
                densities.append(d)

        median_h = statistics.median(heights) if heights else 20
        min_x = min(x_positions) if x_positions else 0
        median_d = statistics.median(densities) if densities else 0.5

        # 2. 라인별 속성 판별
        segments = []
        for line in ocr_result:
            box = line[0]
            raw_text = line[1][0]
            h, x, y, w, pts = self.get_geometry(box)
            
            # 오타 교정
            text = self.fix_typos(raw_text)
            
            # 속성 판별
            ratio = h / median_h
            is_indented = (x - min_x) >= self.INDENT_PX
            is_bold = self.is_bold(pts, median_d)
            
            # 태그 결정
            tag = "p"
            if ratio >= self.H1_SCALE: tag = "h1"
            elif ratio >= self.H2_SCALE: tag = "h2"
            
            segments.append({
                "category": "text",
                "y_pos": y,       # 문서 내 순서 정렬용
                "text": text,
                "tag": tag,
                "is_bold": is_bold,
                "is_indented": is_indented
            })
        return segments

# =========================================================
# 2. HybridOCRPipeline: 실제 모델 구동 및 실행 클래스
# =========================================================
class HybridOCRPipeline:
    """
    [통합 파이프라인]
    LayoutAnalyzer(YOLO) -> PaddleOCR(Text) + GOT-OCR(Table) -> UltimateLegalParser
    """
    def __init__(self, got_model_path, yolo_model_path):
        # 1. Layout Analyzer 로드 (표 분리)
        if LayoutAnalyzer:
            self.layout_analyzer = LayoutAnalyzer(yolo_model_path)
        else:
            raise ImportError("LayoutAnalyzer 클래스를 불러올 수 없습니다.")
        
        # 2. PaddleOCR 로드 (한글 텍스트용)
        # GPU 사용 시 use_gpu=True, CPU면 False
        print("Loading PaddleOCR...")
        self.paddle = PaddleOCR(lang='korean', use_angle_cls=True, show_log=False, use_gpu=True)
        
        # 3. GOT-OCR 로드 (표 인식용)
        print("Loading GOT-OCR model... (이 작업은 시간이 걸립니다)")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(got_model_path, trust_remote_code=True)
            self.got_model = AutoModelForCausalLM.from_pretrained(
                got_model_path, 
                trust_remote_code=True, 
                low_cpu_mem_usage=True,
                device_map='cuda', # GPU 필수 권장
                use_safetensors=True
            ).eval().cuda()
        except Exception as e:
            print(f"[Critical] GOT-OCR Load Failed: {e}")
            self.got_model = None

    def run(self, image_path):
        print(f"Processing Image: {image_path}")
        
        # Step 1: 레이아웃 분리
        # tables: 표 이미지 리스트, text_image_bgr: 표가 지워진(흰색 칠해진) 텍스트 이미지
        tables, text_image_bgr = self.layout_analyzer.split_content(image_path)
        
        # Grayscale 변환 (UltimateParser 볼드체 분석용)
        text_image_gray = cv2.cvtColor(text_image_bgr, cv2.COLOR_BGR2GRAY)

        # Step 2: 텍스트 OCR (Paddle)
        paddle_results = self.paddle.ocr(text_image_bgr, cls=True)
        
        # 텍스트 구조화 분석 (UltimateLegalParser)
        text_parser = UltimateLegalParser(text_image_gray)
        structured_texts = []
        if paddle_results and paddle_results[0]:
             structured_texts = text_parser.parse_text_segments(paddle_results[0])

        # Step 3: 표 OCR (GOT-OCR)
        structured_tables = []
        for tbl in tables:
            box = tbl['box'] # [x1, y1, x2, y2]
            y_pos = box[1]   # Y좌표 (나중에 텍스트와 순서 섞기 위해 필요)
            
            # GOT-OCR은 파일 경로 입력을 선호하므로 임시 저장
            temp_tbl_path = f"temp_table_{y_pos}.png"
            cv2.imwrite(temp_tbl_path, tbl['image'])
            
            res = "[표 인식 불가]"
            if self.got_model:
                try:
                    # ocr_type='format' : 마크다운/HTML 형식 출력 모드
                    res = self.got_model.chat(self.tokenizer, temp_tbl_path, ocr_type='format')
                except Exception as e:
                    print(f"GOT Error: {e}")
                    res = "[표 인식 중 오류 발생]"
            
            structured_tables.append({
                "category": "table",
                "y_pos": y_pos,
                "text": res
            })
            # 임시 파일 삭제
            if os.path.exists(temp_tbl_path): os.remove(temp_tbl_path)

        # Step 4: 결과 합치기 (Y좌표 기준 정렬)
        # 텍스트와 표를 위에서 아래 순서대로 섞음
        all_segments = structured_texts + structured_tables
        all_segments.sort(key=lambda x: x['y_pos'])

        # Step 5: 최종 마크다운 생성
        final_md = []
        for seg in all_segments:
            if seg['category'] == 'table':
                # 표는 앞뒤로 줄바꿈 추가
                final_md.append("\n" + seg['text'] + "\n")
            else:
                # 텍스트 포맷팅 적용
                prefix = ""
                # 헤더 태그 변환
                if seg['tag'] == "h1": prefix = "# "
                elif seg['tag'] == "h2": prefix = "## "
                
                # 들여쓰기 변환
                if seg['is_indented']: prefix = "> " + prefix
                
                text = seg['text']
                # 볼드체 변환
                if seg['is_bold']: text = f"**{text}**"
                
                final_md.append(f"{prefix}{text}")

        # 문단 간격 조정을 위해 두 줄 띄우기 결합
        return "\n\n".join(final_md)