import cv2
import numpy as np
from ultralytics import YOLO

class LayoutAnalyzer:
    def __init__(self, model_path='yolov8n.pt'):
        """
        초기화: YOLO 모델 로드
        model_path: 표 탐지에 특화된 모델 경로 (.pt) 권장
        (예: 'foduucom/table-detection-and-extraction'의 가중치)
        """
        self.model_path = model_path
        try:
            # 커스텀 모델이 없으면 기본 모델 로드 (경고 발생 가능)
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[Warning] 지정된 모델 로드 실패, 기본 yolov8n.pt 사용: {e}")
            self.model = YOLO('yolov8n.pt')
            
        self.TABLE_CLASS_ID = 0  # 모델에 따라 다를 수 있음 (보통 0)

    def detect_borderless_tables(self, image, existing_boxes):
        """
        [알고리즘] YOLO가 놓친 '선 없는 표'를 수직 공백(Vertical River) 분석으로 탐지
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 이진화 및 전처리
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 2. 텍스트 라인 뭉치기 (수평 팽창)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        dilated = cv2.dilate(binary, kernel_h, iterations=1)
        
        # 3. 윤곽선 검출
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_tables = []
        img_h, img_w = image.shape[:2]
        min_area = (img_w * img_h) * 0.05  # 전체의 5% 이상 크기만

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 기존 YOLO 박스와 겹치면 패스
            is_overlap = False
            for (ex_x1, ex_y1, ex_x2, ex_y2) in existing_boxes:
                # IOU 비슷한 로직: 교차 영역이 있으면 건너뜀
                if not (x > ex_x2 or x + w < ex_x1 or y > ex_y2 or y + h < ex_y1):
                    is_overlap = True
                    break
            if is_overlap or (w * h < min_area): continue

            # 4. 수직 투영 (Vertical Projection)으로 공백 확인
            roi = binary[y:y+h, x:x+w]
            v_proj = np.sum(roi, axis=0)
            
            # 값이 0인 구간(공백) 찾기
            blank_indices = np.where(v_proj == 0)[0]
            
            if len(blank_indices) > 0:
                # 공백 구간이 끊어지는 지점을 찾아 '열(Column)' 개수 추정
                gaps = np.diff(blank_indices)
                # gap이 1이 아닌 지점이 열과 열 사이의 텍스트 덩어리
                num_columns = len(np.where(gaps > 1)[0]) + 1
                
                # 열이 2개 이상이면 '선 없는 표'로 간주
                if num_columns >= 2:
                    potential_tables.append([x, y, x+w, y+h])

        return potential_tables

    def split_content(self, image_path):
        """
        이미지를 입력받아 [표 리스트]와 [텍스트만 남은 이미지]로 분리 반환
        """
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Image load failed: {image_path}")
        
        tables = []
        detected_boxes = [] # [x1, y1, x2, y2]

        # 1. YOLO 실행 (1차 탐지)
        results = self.model(image)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == self.TABLE_CLASS_ID and conf > 0.4: # 신뢰도 0.4 이상
                    detected_boxes.append([x1, y1, x2, y2])
                    
                    # 표 이미지 잘라내기 (Deep Copy)
                    crop = image[y1:y2, x1:x2].copy()
                    tables.append({
                        "type": "yolo_table",
                        "box": [x1, y1, x2, y2],
                        "image": crop
                    })
                    # 원본에서 지우기 (흰색 칠하기) -> PaddleOCR이 읽지 못하게
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # 2. Heuristic 실행 (2차 탐지 - 선 없는 표)
        hidden_tables = self.detect_borderless_tables(image, detected_boxes)
        for (x1, y1, x2, y2) in hidden_tables:
            crop = image[y1:y2, x1:x2].copy()
            tables.append({
                "type": "heuristic_table",
                "box": [x1, y1, x2, y2],
                "image": crop
            })
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # tables: GOT-OCR로 보낼 이미지 리스트
        # image: 텍스트만 남은 이미지 (PaddleOCR용)
        return tables, image