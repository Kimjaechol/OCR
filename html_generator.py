"""
Legal Document OCR - HTML Generator Module
===========================================
Converts OCR results to formatted HTML with preserved styling
Supports headings, bold text, tables, and document structure
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import markdownify
    MARKDOWNIFY_AVAILABLE = True
except ImportError:
    logger.warning("markdownify not installed. HTML to Markdown conversion unavailable.")
    MARKDOWNIFY_AVAILABLE = False


class TextStyle(Enum):
    """Text styling types"""
    NORMAL = "normal"
    HEADING1 = "h1"
    HEADING2 = "h2"
    HEADING3 = "h3"
    BOLD = "bold"
    TABLE = "table"
    INVISIBLE_TABLE = "invisible_table"  # Table with no visible borders


class TextAlignment(Enum):
    """Text alignment types"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class StyledText:
    """Text segment with style information"""
    text: str
    style: TextStyle
    y_pos: int
    x_pos: int = 0
    font_size: float = 12.0
    is_bold: bool = False
    confidence: float = 1.0
    alignment: str = "left"  # left, center, right
    line_spacing_before: int = 0  # Pixels of empty space before this text
    page_width: int = 0  # Page width for relative positioning


@dataclass
class HTMLDocument:
    """Generated HTML document"""
    html: str
    markdown: str
    title: str = ""
    page_count: int = 1


class HTMLGenerator:
    """
    Generates formatted HTML from OCR results.

    Features:
    - Heading detection based on font size
    - Bold text detection
    - Table preservation
    - Clean semantic HTML output
    - HTML to Markdown conversion
    """

    # Font size thresholds for heading detection (relative to median)
    HEADING1_THRESHOLD = 1.8  # 180% of median = H1
    HEADING2_THRESHOLD = 1.4  # 140% of median = H2
    HEADING3_THRESHOLD = 1.2  # 120% of median = H3

    def __init__(self):
        """Initialize HTML generator"""
        self.css_styles = self._get_default_css()

    def _get_default_css(self) -> str:
        """Get default CSS styles for HTML output"""
        return """
        <style>
            body {
                font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
                line-height: 1.8;
                max-width: 900px;
                margin: 0 auto;
                padding: 40px 20px;
                color: #333;
                background: #fff;
            }
            h1 {
                font-size: 24px;
                font-weight: bold;
                margin: 30px 0 15px 0;
                color: #1a1a1a;
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }
            h2 {
                font-size: 20px;
                font-weight: bold;
                margin: 25px 0 12px 0;
                color: #2a2a2a;
            }
            h3 {
                font-size: 17px;
                font-weight: bold;
                margin: 20px 0 10px 0;
                color: #3a3a3a;
            }
            p {
                margin: 10px 0;
                text-align: justify;
            }
            /* Text alignment classes */
            .text-left {
                text-align: left;
            }
            .text-center {
                text-align: center;
            }
            .text-right {
                text-align: right;
            }
            /* Empty line / spacing */
            .empty-line {
                height: 1em;
                margin: 0;
            }
            .spacing-small {
                margin-top: 0.5em;
            }
            .spacing-medium {
                margin-top: 1em;
            }
            .spacing-large {
                margin-top: 2em;
            }
            strong, b {
                font-weight: bold;
            }
            /* Visible table with borders */
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }
            table.visible-table th,
            table.visible-table td {
                border: 1px solid #ccc;
                padding: 10px 12px;
                text-align: left;
                vertical-align: top;
            }
            table.visible-table th {
                background-color: #f5f5f5;
                font-weight: bold;
            }
            table.visible-table tr:nth-child(even) {
                background-color: #fafafa;
            }
            /* Invisible table (no borders) for government forms */
            table.invisible-table {
                border: none;
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }
            table.invisible-table th,
            table.invisible-table td {
                border: none;
                padding: 5px 10px;
                text-align: left;
                vertical-align: top;
            }
            table.invisible-table th {
                background-color: transparent;
                font-weight: normal;
            }
            /* Preserve exact positioning */
            .positioned-text {
                position: relative;
                white-space: pre-wrap;
            }
            .page-break {
                page-break-after: always;
                border-top: 1px dashed #ccc;
                margin: 40px 0;
                padding-top: 20px;
            }
            .page-header {
                color: #888;
                font-size: 12px;
                margin-bottom: 20px;
            }
            blockquote {
                border-left: 4px solid #ddd;
                padding-left: 20px;
                margin: 15px 0;
                color: #555;
            }
            .legal-clause {
                margin: 15px 0;
                padding: 10px 15px;
                background: #f9f9f9;
                border-radius: 4px;
            }
            /* Font size variations */
            .font-small { font-size: 11px; }
            .font-normal { font-size: 14px; }
            .font-large { font-size: 18px; }
            .font-xlarge { font-size: 22px; }
            /* Print styles */
            @media print {
                body {
                    max-width: none;
                    padding: 0;
                }
                .page-break {
                    page-break-after: always;
                    border: none;
                    margin: 0;
                    padding: 0;
                }
                table.invisible-table th,
                table.invisible-table td {
                    border: none !important;
                }
            }
        </style>
        """

    def detect_text_style(
        self,
        text: str,
        font_size: float,
        median_font_size: float,
        is_bold: bool = False,
        line_count: int = 1
    ) -> TextStyle:
        """
        Detect text style based on font size and other characteristics.

        Args:
            text: The text content
            font_size: Detected font size
            median_font_size: Median font size of the document
            is_bold: Whether text appears bold
            line_count: Number of lines in the text block

        Returns:
            TextStyle enum value
        """
        if median_font_size <= 0:
            median_font_size = 12.0

        size_ratio = font_size / median_font_size

        # Short text with large font = likely heading
        is_short = len(text.strip()) < 100 and line_count <= 2

        if is_short:
            if size_ratio >= self.HEADING1_THRESHOLD:
                return TextStyle.HEADING1
            elif size_ratio >= self.HEADING2_THRESHOLD:
                return TextStyle.HEADING2
            elif size_ratio >= self.HEADING3_THRESHOLD or is_bold:
                return TextStyle.HEADING3

        # Legal document patterns for headings
        heading_patterns = [
            r'^제\s*\d+\s*조',           # 제1조, 제 2 조
            r'^제\s*\d+\s*장',           # 제1장
            r'^제\s*\d+\s*편',           # 제1편
            r'^제\s*\d+\s*절',           # 제1절
            r'^\d+\.\s+[가-힣]',         # 1. 가나다
            r'^[一二三四五六七八九十]+[、\.]', # 一. 二. (한자 번호)
            r'^[①②③④⑤⑥⑦⑧⑨⑩]',      # 원문자 번호
            r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[\.、]',  # 로마 숫자
        ]

        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                if size_ratio >= self.HEADING2_THRESHOLD:
                    return TextStyle.HEADING1
                elif size_ratio >= self.HEADING3_THRESHOLD:
                    return TextStyle.HEADING2
                else:
                    return TextStyle.HEADING3

        if is_bold and is_short:
            return TextStyle.BOLD

        return TextStyle.NORMAL

    def generate_html(
        self,
        segments: List[StyledText],
        title: str = "OCR Document",
        include_css: bool = True
    ) -> str:
        """
        Generate HTML from styled text segments with alignment and spacing.

        Args:
            segments: List of StyledText objects
            title: Document title
            include_css: Whether to include CSS styles

        Returns:
            Complete HTML document string
        """
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="ko">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'<title>{self._escape_html(title)}</title>',
        ]

        if include_css:
            html_parts.append(self.css_styles)

        html_parts.extend([
            '</head>',
            '<body>',
        ])

        prev_segment = None

        for segment in segments:
            # Add spacing/empty lines before this segment
            spacing_html = self._create_spacing(segment, prev_segment)
            if spacing_html:
                html_parts.append(spacing_html)

            if segment.style == TextStyle.TABLE:
                # Visible table with borders
                html_parts.append(self._wrap_table_with_class(segment.text, "visible-table"))

            elif segment.style == TextStyle.INVISIBLE_TABLE:
                # Invisible table without borders
                html_parts.append(self._wrap_table_with_class(segment.text, "invisible-table"))

            elif segment.style in (TextStyle.HEADING1, TextStyle.HEADING2, TextStyle.HEADING3):
                html_parts.append(self._create_heading(segment))

            elif segment.style == TextStyle.BOLD:
                html_parts.append(self._create_styled_paragraph(
                    f'<strong>{self._escape_html(segment.text)}</strong>',
                    segment
                ))

            else:
                html_parts.append(self._create_styled_paragraph(
                    self._escape_html(segment.text),
                    segment
                ))

            prev_segment = segment

        html_parts.extend([
            '</body>',
            '</html>'
        ])

        return '\n'.join(html_parts)

    def _create_spacing(self, segment: StyledText, prev_segment: Optional[StyledText]) -> str:
        """
        Create HTML for spacing/empty lines between segments.

        Args:
            segment: Current segment
            prev_segment: Previous segment (or None)

        Returns:
            HTML string for spacing (empty if no spacing needed)
        """
        if prev_segment is None:
            return ''

        spacing = segment.line_spacing_before

        # Estimate typical line height (in pixels)
        typical_line_height = 25

        if spacing <= typical_line_height:
            return ''
        elif spacing <= typical_line_height * 2:
            return '<div class="empty-line"></div>'
        elif spacing <= typical_line_height * 3:
            return '<div class="empty-line"></div><div class="empty-line"></div>'
        else:
            # Multiple empty lines
            num_lines = min(spacing // typical_line_height, 5)
            return ''.join(['<div class="empty-line"></div>'] * num_lines)

    def _create_heading(self, segment: StyledText) -> str:
        """Create heading HTML element with alignment"""
        tag = segment.style.value  # h1, h2, or h3
        text = self._escape_html(segment.text)
        alignment = segment.alignment

        if alignment == "center":
            return f'<{tag} class="text-center">{text}</{tag}>'
        elif alignment == "right":
            return f'<{tag} class="text-right">{text}</{tag}>'
        else:
            return f'<{tag}>{text}</{tag}>'

    def _create_styled_paragraph(self, content: str, segment: StyledText) -> str:
        """
        Create paragraph with alignment and font size.

        Args:
            content: The HTML content (already escaped)
            segment: StyledText with formatting info

        Returns:
            HTML paragraph string
        """
        if not content.strip():
            return ''

        classes = []
        styles = []

        # Add alignment class
        if segment.alignment == "center":
            classes.append("text-center")
        elif segment.alignment == "right":
            classes.append("text-right")

        # Add font size class based on estimated size
        if segment.font_size > 20:
            classes.append("font-xlarge")
        elif segment.font_size > 16:
            classes.append("font-large")
        elif segment.font_size < 10:
            classes.append("font-small")

        # Build the tag
        class_attr = f' class="{" ".join(classes)}"' if classes else ''
        style_attr = f' style="{"; ".join(styles)}"' if styles else ''

        return f'<p{class_attr}{style_attr}>{content}</p>'

    def _wrap_table_with_class(self, table_html: str, css_class: str) -> str:
        """
        Add CSS class to table element.

        Args:
            table_html: HTML table string
            css_class: CSS class to add

        Returns:
            Modified table HTML
        """
        # If table already has class, append to it
        if 'class="' in table_html:
            return table_html.replace('class="', f'class="{css_class} ')
        else:
            return table_html.replace('<table>', f'<table class="{css_class}">')

    def _create_paragraph(self, parts: List[str], alignment: str = "left") -> str:
        """Create paragraph from text parts with alignment"""
        content = ' '.join(parts)
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        if content:
            if alignment == "center":
                return f'<p class="text-center">{content}</p>'
            elif alignment == "right":
                return f'<p class="text-right">{content}</p>'
            return f'<p>{content}</p>'
        return ''

    def generate_invisible_table_html(
        self,
        cells: List[Dict],
        rows: int,
        cols: int
    ) -> str:
        """
        Generate HTML for invisible table (government forms).

        Args:
            cells: List of cell dicts with 'text', 'row', 'col'
            rows: Number of rows
            cols: Number of columns

        Returns:
            HTML table string with no visible borders
        """
        # Create empty grid
        grid = [['' for _ in range(cols)] for _ in range(rows)]

        # Fill in cells
        for cell in cells:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            text = cell.get('text', '')
            if 0 <= row < rows and 0 <= col < cols:
                grid[row][col] = text

        # Generate HTML
        html_parts = ['<table class="invisible-table">']

        for row in grid:
            html_parts.append('<tr>')
            for cell_text in row:
                escaped = self._escape_html(cell_text)
                html_parts.append(f'<td>{escaped}</td>')
            html_parts.append('</tr>')

        html_parts.append('</table>')

        return '\n'.join(html_parts)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))

    def markdown_table_to_html(self, markdown_table: str) -> str:
        """
        Convert markdown table to HTML table.

        Args:
            markdown_table: Markdown formatted table

        Returns:
            HTML table string
        """
        lines = markdown_table.strip().split('\n')
        if len(lines) < 2:
            return f'<p>{self._escape_html(markdown_table)}</p>'

        html_parts = ['<table>']
        header_added = False
        tbody_opened = False

        for i, line in enumerate(lines):
            # Skip separator line (---)
            if re.match(r'^[\s\|:-]+$', line):
                continue

            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]  # Remove empty cells from edges

            if not cells:
                continue

            if not header_added:
                # First valid row becomes header
                html_parts.append('<thead><tr>')
                for cell in cells:
                    html_parts.append(f'<th>{self._escape_html(cell)}</th>')
                html_parts.append('</tr></thead>')
                html_parts.append('<tbody>')
                header_added = True
                tbody_opened = True
            else:
                # Data row
                html_parts.append('<tr>')
                for cell in cells:
                    html_parts.append(f'<td>{self._escape_html(cell)}</td>')
                html_parts.append('</tr>')

        if tbody_opened:
            html_parts.append('</tbody>')
        html_parts.append('</table>')

        return '\n'.join(html_parts)

    def html_to_markdown(self, html: str) -> str:
        """
        Convert HTML to Markdown.

        Args:
            html: HTML string

        Returns:
            Markdown string
        """
        if not MARKDOWNIFY_AVAILABLE:
            logger.warning("markdownify not available, returning stripped HTML")
            return self._strip_html_tags(html)

        try:
            md = markdownify.markdownify(
                html,
                heading_style="ATX",  # Use # style headings
                bullets="-",
                strip=['script', 'style']
            )

            # Clean up excessive newlines
            md = re.sub(r'\n{3,}', '\n\n', md)

            return md.strip()

        except Exception as e:
            logger.error(f"HTML to Markdown conversion error: {e}")
            return self._strip_html_tags(html)

    def _strip_html_tags(self, html: str) -> str:
        """Remove HTML tags (fallback when markdownify unavailable)"""
        # Remove style and script content
        text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Convert some tags to markdown-like format
        text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\n# \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n## \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

        # Remove remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def generate_document(
        self,
        segments: List[StyledText],
        title: str = "OCR Document"
    ) -> HTMLDocument:
        """
        Generate both HTML and Markdown from styled segments.

        Args:
            segments: List of StyledText objects
            title: Document title

        Returns:
            HTMLDocument with both formats
        """
        html = self.generate_html(segments, title)
        markdown = self.html_to_markdown(html)

        return HTMLDocument(
            html=html,
            markdown=markdown,
            title=title,
            page_count=1
        )


class DocumentFormatter:
    """
    Formats OCR results into styled documents.
    Bridges OCR output to HTML generator.
    """

    def __init__(self):
        self.html_generator = HTMLGenerator()

    def format_ocr_results(
        self,
        text_segments: List[Dict],
        table_segments: List[Dict],
        page_number: int = 1,
        page_width: int = 0,
        invisible_tables: List[Dict] = None
    ) -> HTMLDocument:
        """
        Format OCR results into HTML and Markdown.

        Args:
            text_segments: List of text segment dicts with 'text', 'bbox', 'confidence',
                          'alignment', 'line_spacing_before'
            table_segments: List of table segment dicts with 'markdown', 'bbox'
            page_number: Page number
            page_width: Page width for relative positioning
            invisible_tables: List of invisible table dicts for government forms

        Returns:
            HTMLDocument with formatted content
        """
        styled_segments = []

        # Calculate median font size from bounding boxes
        font_sizes = []
        for seg in text_segments:
            if 'bbox' in seg and len(seg['bbox']) >= 4:
                height = seg['bbox'][3] - seg['bbox'][1]
                font_sizes.append(height)

        median_font_size = sorted(font_sizes)[len(font_sizes)//2] if font_sizes else 20

        # Process text segments
        for seg in text_segments:
            text = seg.get('text', '')
            bbox = seg.get('bbox', [0, 0, 0, 0])
            confidence = seg.get('confidence', 1.0)
            is_bold = seg.get('is_bold', False)
            alignment = seg.get('alignment', 'left')
            line_spacing = seg.get('line_spacing_before', 0)

            if len(bbox) >= 4:
                font_size = bbox[3] - bbox[1]  # Height as proxy for font size
                y_pos = bbox[1]
                x_pos = bbox[0]
            else:
                font_size = median_font_size
                y_pos = 0
                x_pos = 0

            style = self.html_generator.detect_text_style(
                text=text,
                font_size=font_size,
                median_font_size=median_font_size,
                is_bold=is_bold
            )

            styled_segments.append(StyledText(
                text=text,
                style=style,
                y_pos=y_pos,
                x_pos=x_pos,
                font_size=font_size,
                is_bold=is_bold,
                confidence=confidence,
                alignment=alignment,
                line_spacing_before=line_spacing,
                page_width=page_width
            ))

        # Process visible table segments
        for seg in table_segments:
            markdown_table = seg.get('markdown', '')
            bbox = seg.get('bbox', [0, 0, 0, 0])
            y_pos = bbox[1] if len(bbox) >= 2 else 0
            is_invisible = seg.get('is_invisible', False)

            # Convert markdown table to HTML
            html_table = self.html_generator.markdown_table_to_html(markdown_table)

            styled_segments.append(StyledText(
                text=html_table,
                style=TextStyle.INVISIBLE_TABLE if is_invisible else TextStyle.TABLE,
                y_pos=y_pos,
                x_pos=0,
                font_size=0,
                is_bold=False
            ))

        # Process invisible tables (government forms)
        if invisible_tables:
            for inv_table in invisible_tables:
                cells = inv_table.get('cells', [])
                rows = inv_table.get('rows', 0)
                cols = inv_table.get('cols', 0)
                bbox = inv_table.get('bbox', [0, 0, 0, 0])
                y_pos = bbox[1] if len(bbox) >= 2 else 0

                if rows > 0 and cols > 0:
                    html_table = self.html_generator.generate_invisible_table_html(
                        cells=cells,
                        rows=rows,
                        cols=cols
                    )

                    styled_segments.append(StyledText(
                        text=html_table,
                        style=TextStyle.INVISIBLE_TABLE,
                        y_pos=y_pos,
                        x_pos=0,
                        font_size=0,
                        is_bold=False
                    ))

        # Sort by Y position
        styled_segments.sort(key=lambda x: x.y_pos)

        return self.html_generator.generate_document(
            segments=styled_segments,
            title=f"OCR Document - Page {page_number}"
        )
