"""
Legal Document OCR - Layout Analyzer Module
============================================
YOLOv8 + Heuristic table detection for legal documents
Detects both bordered and borderless tables
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed. YOLO detection will be unavailable.")
    YOLO = None


@dataclass
class DetectedRegion:
    """Represents a detected region (table or text area)"""
    region_type: str  # 'yolo_table', 'heuristic_table', 'text'
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    image: Optional[np.ndarray] = None


class TextAlignment(Enum):
    """Text alignment types"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class TextFormatInfo:
    """Text formatting information detected from image analysis"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    estimated_font_size: float
    is_bold: bool
    is_heading: bool
    stroke_width: float = 1.0
    alignment: str = "left"  # left, center, right
    line_spacing_before: int = 0  # pixels of empty space before this text


@dataclass
class InvisibleTableCell:
    """Cell in an invisible table"""
    bbox: Tuple[int, int, int, int]
    text: str = ""
    row: int = 0
    col: int = 0


@dataclass
class InvisibleTableInfo:
    """Detected invisible table structure"""
    bbox: Tuple[int, int, int, int]
    rows: int
    cols: int
    cells: List[InvisibleTableCell] = field(default_factory=list)
    column_positions: List[int] = field(default_factory=list)
    row_positions: List[int] = field(default_factory=list)


class LayoutAnalyzer:
    """
    Layout analyzer for legal documents using YOLOv8 and heuristic methods.

    Features:
    - YOLO-based table detection (bordered tables)
    - Heuristic-based borderless table detection (vertical river analysis)
    - Configurable confidence thresholds
    - GPU acceleration support
    """

    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        confidence_threshold: float = 0.4,
        table_class_id: int = 0,
        min_area_ratio: float = 0.05,
        use_gpu: bool = True
    ):
        """
        Initialize the layout analyzer.

        Args:
            model_path: Path to YOLO model (.pt file)
            confidence_threshold: Minimum confidence for table detection
            table_class_id: Class ID for tables in the model
            min_area_ratio: Minimum area ratio for borderless table detection
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.table_class_id = table_class_id
        self.min_area_ratio = min_area_ratio
        self.use_gpu = use_gpu
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model with fallback handling"""
        if YOLO is None:
            logger.error("YOLO not available. Table detection disabled.")
            return

        try:
            model_path = Path(self.model_path)
            if model_path.exists():
                self.model = YOLO(str(model_path))
                logger.info(f"Loaded YOLO model from: {model_path}")
            else:
                # Try default model
                logger.warning(f"Model not found at {model_path}, using yolov8n.pt")
                self.model = YOLO('yolov8n.pt')

            # Set device
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.model.to('cuda')
                        logger.info("YOLO model loaded on GPU")
                    else:
                        logger.info("CUDA not available, using CPU")
                except Exception as e:
                    logger.warning(f"Could not move model to GPU: {e}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _boxes_overlap(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
        threshold: float = 0.1
    ) -> bool:
        """Check if two boxes overlap significantly"""
        x1, y1, x2, y2 = box1
        ex1, ey1, ex2, ey2 = box2

        # Check for any overlap
        if x1 > ex2 or x2 < ex1 or y1 > ey2 or y2 < ey1:
            return False

        return self._calculate_iou(box1, box2) > threshold

    def detect_tables_yolo(
        self,
        image: np.ndarray
    ) -> List[DetectedRegion]:
        """
        Detect tables using YOLO model.

        Args:
            image: BGR image array

        Returns:
            List of detected table regions
        """
        tables = []

        if self.model is None:
            logger.warning("YOLO model not loaded, skipping YOLO detection")
            return tables

        try:
            results = self.model(image, verbose=False)

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    # Filter by class and confidence
                    if cls == self.table_class_id and conf >= self.confidence_threshold:
                        # Ensure valid coordinates
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(image.shape[1], x2)
                        y2 = min(image.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            crop = image[y1:y2, x1:x2].copy()
                            tables.append(DetectedRegion(
                                region_type="yolo_table",
                                box=(x1, y1, x2, y2),
                                confidence=conf,
                                image=crop
                            ))

            logger.debug(f"YOLO detected {len(tables)} tables")

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")

        return tables

    def detect_borderless_tables(
        self,
        image: np.ndarray,
        existing_boxes: List[Tuple[int, int, int, int]]
    ) -> List[DetectedRegion]:
        """
        Detect borderless tables using vertical projection analysis.

        Algorithm:
        1. Binarize and dilate horizontally to merge text lines
        2. Find contours of text blocks
        3. Analyze vertical projection to find column gaps
        4. If multiple columns detected, mark as borderless table

        Args:
            image: BGR image array
            existing_boxes: List of already detected table boxes to avoid overlap

        Returns:
            List of detected borderless table regions
        """
        tables = []

        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            img_h, img_w = gray.shape[:2]
            min_area = (img_w * img_h) * self.min_area_ratio

            # Binary threshold
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # Horizontal dilation to merge text lines
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            dilated = cv2.dilate(binary, kernel_h, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Skip small regions
                if w * h < min_area:
                    continue

                # Skip if overlaps with existing tables
                current_box = (x, y, x + w, y + h)
                is_overlap = any(
                    self._boxes_overlap(current_box, existing_box)
                    for existing_box in existing_boxes
                )
                if is_overlap:
                    continue

                # Vertical projection analysis
                roi = binary[y:y+h, x:x+w]
                v_proj = np.sum(roi, axis=0)

                # Find blank columns (gaps)
                blank_indices = np.where(v_proj == 0)[0]

                if len(blank_indices) > 0:
                    # Count column separators
                    gaps = np.diff(blank_indices)
                    num_columns = len(np.where(gaps > 1)[0]) + 1

                    # If 2+ columns, likely a borderless table
                    if num_columns >= 2:
                        # Ensure valid coordinates
                        x1, y1 = max(0, x), max(0, y)
                        x2 = min(img_w, x + w)
                        y2 = min(img_h, y + h)

                        crop = image[y1:y2, x1:x2].copy()
                        tables.append(DetectedRegion(
                            region_type="heuristic_table",
                            box=(x1, y1, x2, y2),
                            confidence=0.7,  # Lower confidence for heuristic
                            image=crop
                        ))

            logger.debug(f"Heuristic detected {len(tables)} borderless tables")

        except Exception as e:
            logger.error(f"Borderless table detection error: {e}")

        return tables

    def split_content(
        self,
        image_input: Union[str, np.ndarray, Path]
    ) -> Tuple[List[DetectedRegion], np.ndarray]:
        """
        Split image into tables and text regions.

        Args:
            image_input: Image path or BGR image array

        Returns:
            Tuple of (list of table regions, text-only image)
        """
        # Load image if path provided
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Failed to load image: {image_input}")
        else:
            image = image_input.copy()

        all_tables = []
        detected_boxes = []

        # Step 1: YOLO detection (bordered tables)
        yolo_tables = self.detect_tables_yolo(image)
        for table in yolo_tables:
            all_tables.append(table)
            detected_boxes.append(table.box)

            # Mask out table region (white fill)
            x1, y1, x2, y2 = table.box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Step 2: Heuristic detection (borderless tables)
        heuristic_tables = self.detect_borderless_tables(image, detected_boxes)
        for table in heuristic_tables:
            all_tables.append(table)

            # Mask out table region
            x1, y1, x2, y2 = table.box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        logger.info(
            f"Layout analysis complete: {len(yolo_tables)} YOLO tables, "
            f"{len(heuristic_tables)} heuristic tables"
        )

        return all_tables, image

    def detect_text_formatting(
        self,
        image: np.ndarray,
        text_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[TextFormatInfo]:
        """
        Detect text formatting (bold, heading, alignment, spacing) using OpenCV analysis.

        Algorithm:
        1. For each text bbox, calculate stroke width using distance transform
        2. Compare stroke width to median to detect bold
        3. Use bbox height to estimate font size and detect headings
        4. Analyze x position relative to page width to detect alignment
        5. Calculate vertical spacing between text blocks

        Args:
            image: BGR image array
            text_bboxes: List of text bounding boxes [(x1,y1,x2,y2), ...]

        Returns:
            List of TextFormatInfo for each bbox
        """
        format_infos = []

        if len(text_bboxes) == 0:
            return format_infos

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        img_h, img_w = gray.shape[:2]

        # Calculate stroke widths for all bboxes
        stroke_widths = []
        heights = []

        for bbox in text_bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(gray.shape[1], x2)
            y2 = min(gray.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                stroke_widths.append(1.0)
                heights.append(20)
                continue

            roi = gray[y1:y2, x1:x2]
            height = y2 - y1
            heights.append(height)

            # Calculate stroke width using distance transform
            stroke_width = self._calculate_stroke_width(roi)
            stroke_widths.append(stroke_width)

        # Calculate medians for comparison
        median_stroke = np.median(stroke_widths) if stroke_widths else 1.0
        median_height = np.median(heights) if heights else 20

        # Bold threshold: stroke width > 1.3x median
        bold_threshold = median_stroke * 1.3

        # Heading threshold: height > 1.2x median
        heading_threshold = median_height * 1.2

        # Sort bboxes by Y position for line spacing calculation
        sorted_indices = sorted(range(len(text_bboxes)), key=lambda i: text_bboxes[i][1])

        # Calculate alignments for all bboxes
        alignments = self._detect_text_alignments(text_bboxes, img_w)

        for i, bbox in enumerate(text_bboxes):
            x1, y1, x2, y2 = bbox
            stroke_width = stroke_widths[i]
            height = heights[i]

            is_bold = stroke_width > bold_threshold
            is_heading = height > heading_threshold
            alignment = alignments[i]

            # Calculate line spacing before this text
            line_spacing = self._calculate_line_spacing(i, text_bboxes, sorted_indices)

            format_infos.append(TextFormatInfo(
                bbox=bbox,
                estimated_font_size=height * 0.75,  # Approximate pt size
                is_bold=is_bold,
                is_heading=is_heading,
                stroke_width=stroke_width,
                alignment=alignment,
                line_spacing_before=line_spacing
            ))

        return format_infos

    def _detect_text_alignments(
        self,
        text_bboxes: List[Tuple[int, int, int, int]],
        page_width: int
    ) -> List[str]:
        """
        Detect text alignment for each bbox based on position.

        Args:
            text_bboxes: List of text bounding boxes
            page_width: Width of the page/image

        Returns:
            List of alignment strings ("left", "center", "right")
        """
        alignments = []

        # Define margins (typically 5-15% of page width)
        left_margin = page_width * 0.08
        right_margin = page_width * 0.92

        # Center detection tolerance (within 10% of center)
        center_tolerance = page_width * 0.10

        for bbox in text_bboxes:
            x1, y1, x2, y2 = bbox
            text_width = x2 - x1
            text_center = (x1 + x2) / 2
            page_center = page_width / 2

            # Check if text spans most of the width (full-width text)
            if text_width > page_width * 0.7:
                alignments.append("left")
                continue

            # Check for center alignment
            if abs(text_center - page_center) < center_tolerance:
                # Additional check: left and right margins should be similar
                left_space = x1
                right_space = page_width - x2
                if abs(left_space - right_space) < page_width * 0.15:
                    alignments.append("center")
                    continue

            # Check for right alignment
            if x2 > right_margin and x1 > page_width * 0.4:
                alignments.append("right")
                continue

            # Default to left alignment
            alignments.append("left")

        return alignments

    def _calculate_line_spacing(
        self,
        current_idx: int,
        text_bboxes: List[Tuple[int, int, int, int]],
        sorted_indices: List[int]
    ) -> int:
        """
        Calculate vertical spacing before a text block.

        Args:
            current_idx: Index of current bbox
            text_bboxes: List of all bboxes
            sorted_indices: Indices sorted by Y position

        Returns:
            Pixel spacing before this text block
        """
        current_pos = sorted_indices.index(current_idx)
        if current_pos == 0:
            return 0

        prev_idx = sorted_indices[current_pos - 1]
        prev_bbox = text_bboxes[prev_idx]
        current_bbox = text_bboxes[current_idx]

        # Calculate gap between previous block's bottom and current block's top
        gap = current_bbox[1] - prev_bbox[3]

        return max(0, gap)

    def detect_invisible_tables(
        self,
        image: np.ndarray,
        text_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[InvisibleTableInfo]:
        """
        Detect invisible tables based on text alignment patterns.

        Government forms often use invisible tables to position text.
        This detects columnar text arrangements that suggest table structure.

        Args:
            image: BGR image array
            text_bboxes: List of text bounding boxes

        Returns:
            List of detected invisible table structures
        """
        invisible_tables = []

        if len(text_bboxes) < 4:
            return invisible_tables

        img_h, img_w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)

        # Group text blocks by their row (similar Y positions)
        row_groups = self._group_by_rows(text_bboxes)

        # Find rows with multiple columns (potential table rows)
        multi_col_rows = [row for row in row_groups if len(row) >= 2]

        if len(multi_col_rows) < 2:
            return invisible_tables

        # Analyze column structure
        column_positions = self._detect_column_positions(multi_col_rows, img_w)

        if len(column_positions) < 2:
            return invisible_tables

        # Find the bounding box of the invisible table
        all_bboxes = [bbox for row in multi_col_rows for bbox in row]
        x1 = min(b[0] for b in all_bboxes)
        y1 = min(b[1] for b in all_bboxes)
        x2 = max(b[2] for b in all_bboxes)
        y2 = max(b[3] for b in all_bboxes)

        # Create cells
        cells = []
        row_positions = []
        for row_idx, row in enumerate(multi_col_rows):
            if row:
                row_positions.append(min(b[1] for b in row))
            for col_idx, bbox in enumerate(sorted(row, key=lambda b: b[0])):
                cells.append(InvisibleTableCell(
                    bbox=bbox,
                    row=row_idx,
                    col=col_idx
                ))

        invisible_tables.append(InvisibleTableInfo(
            bbox=(x1, y1, x2, y2),
            rows=len(multi_col_rows),
            cols=len(column_positions),
            cells=cells,
            column_positions=column_positions,
            row_positions=row_positions
        ))

        return invisible_tables

    def _group_by_rows(
        self,
        text_bboxes: List[Tuple[int, int, int, int]],
        tolerance: float = 0.5
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Group text blocks into rows based on Y position overlap.

        Args:
            text_bboxes: List of text bounding boxes
            tolerance: Overlap tolerance factor

        Returns:
            List of rows, each containing bboxes in that row
        """
        if not text_bboxes:
            return []

        # Sort by Y position
        sorted_bboxes = sorted(text_bboxes, key=lambda b: b[1])

        rows = []
        current_row = [sorted_bboxes[0]]
        current_row_bottom = sorted_bboxes[0][3]

        for bbox in sorted_bboxes[1:]:
            bbox_height = bbox[3] - bbox[1]
            # Check if this bbox overlaps with current row
            if bbox[1] < current_row_bottom - bbox_height * tolerance:
                current_row.append(bbox)
                current_row_bottom = max(current_row_bottom, bbox[3])
            else:
                rows.append(current_row)
                current_row = [bbox]
                current_row_bottom = bbox[3]

        if current_row:
            rows.append(current_row)

        return rows

    def _detect_column_positions(
        self,
        rows: List[List[Tuple[int, int, int, int]]],
        page_width: int
    ) -> List[int]:
        """
        Detect column positions from multiple rows.

        Args:
            rows: List of rows with bboxes
            page_width: Page width for reference

        Returns:
            List of column X positions
        """
        # Collect all x1 positions from multi-column rows
        x_positions = []
        for row in rows:
            sorted_row = sorted(row, key=lambda b: b[0])
            for bbox in sorted_row:
                x_positions.append(bbox[0])

        if not x_positions:
            return []

        # Cluster x positions to find column boundaries
        x_positions.sort()
        clusters = []
        current_cluster = [x_positions[0]]

        for x in x_positions[1:]:
            # If close to last position in cluster, add to cluster
            if x - current_cluster[-1] < page_width * 0.05:
                current_cluster.append(x)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [x]

        if current_cluster:
            clusters.append(int(np.mean(current_cluster)))

        return clusters

    def _calculate_stroke_width(self, roi: np.ndarray) -> float:
        """
        Calculate average stroke width of text in ROI using distance transform.

        Args:
            roi: Grayscale image ROI

        Returns:
            Estimated stroke width in pixels
        """
        try:
            if roi.size == 0:
                return 1.0

            # Binary threshold (text = white on black background for distance transform)
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Distance transform
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

            # Get skeleton for stroke width measurement
            skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else self._simple_skeleton(binary)

            # Measure stroke width at skeleton points
            skeleton_points = np.where(skeleton > 0)

            if len(skeleton_points[0]) == 0:
                # Fallback: use mean of non-zero distance values
                non_zero = dist[dist > 0]
                return float(np.mean(non_zero)) * 2 if len(non_zero) > 0 else 1.0

            stroke_widths = dist[skeleton_points] * 2  # Diameter = 2 * radius
            return float(np.mean(stroke_widths)) if len(stroke_widths) > 0 else 1.0

        except Exception as e:
            logger.debug(f"Stroke width calculation error: {e}")
            return 1.0

    def _simple_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """Simple skeletonization fallback when cv2.ximgproc is not available"""
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()

        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            subset = cv2.subtract(temp, opened)
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()

            if cv2.countNonZero(temp) == 0:
                break

        return skeleton

    def analyze(
        self,
        image_input: Union[str, np.ndarray, Path]
    ) -> Dict:
        """
        Perform full layout analysis.

        Args:
            image_input: Image path or BGR image array

        Returns:
            Dictionary with analysis results
        """
        tables, text_image = self.split_content(image_input)

        return {
            "tables": [
                {
                    "type": t.region_type,
                    "box": t.box,
                    "confidence": t.confidence
                }
                for t in tables
            ],
            "table_count": len(tables),
            "table_images": [t.image for t in tables],
            "text_image": text_image
        }


# Convenience function
def analyze_layout(
    image_path: str,
    model_path: str = 'yolov8n.pt',
    confidence_threshold: float = 0.4
) -> Tuple[List[DetectedRegion], np.ndarray]:
    """
    Convenience function for layout analysis.

    Args:
        image_path: Path to image file
        model_path: Path to YOLO model
        confidence_threshold: Minimum confidence for detection

    Returns:
        Tuple of (table regions, text-only image)
    """
    analyzer = LayoutAnalyzer(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    return analyzer.split_content(image_path)
