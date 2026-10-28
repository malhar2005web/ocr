import os
import shutil
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class BaseOCREngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the OCR engine binary/library is installed and reachable."""
        pass

    @abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        """Extracts text from a PIL Image."""
        pass

class TesseractOCREngine(BaseOCREngine):
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd or "tesseract"
        self._is_ready: Optional[bool] = None

    def _find_binary(self) -> Optional[str]:
        # 1. Check custom configured path
        if self.tesseract_cmd and os.path.exists(self.tesseract_cmd):
            return self.tesseract_cmd
        
        # 2. Check standard Windows installation paths
        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
        ]
        for path in windows_paths:
            if os.path.exists(path):
                return path

        # 3. Check system PATH
        which_path = shutil.which("tesseract")
        if which_path:
            return which_path

        return None

    def is_available(self) -> bool:
        if self._is_ready is not None:
            return self._is_ready

        binary = self._find_binary()
        if binary:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = binary
                self.tesseract_cmd = binary
                self._is_ready = True
                logger.info(f"Tesseract OCR engine ready at: {binary}")
                return True
            except Exception as e:
                logger.warning(f"Failed to initialize pytesseract with {binary}: {e}")
                self._is_ready = False
                return False
        
        logger.info("Tesseract binary not found in standard paths or PATH.")
        self._is_ready = False
        return False

    def preprocess_image(self, image: Any) -> Image.Image:
        """Adaptive image preprocessing: illumination normalization, deskew, and resolution scaling."""
        try:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            np_img = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape

            # 1. Illumination normalization (morphological background division)
            # Flattens shadows, uneven phone camera lighting, and gradient backgrounds
            k_size = max(15, int(min(width, height) / 30))
            if k_size % 2 == 0:
                k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
            norm = cv2.divide(gray, bg, scale=255)

            # 2. Skew detection and correction
            try:
                thresh_skew = cv2.adaptiveThreshold(~norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
                coords = np.column_stack(np.where(thresh_skew > 0))
                if len(coords) > 100:
                    rect = cv2.minAreaRect(coords)
                    angle = rect[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    elif angle > 45:
                        angle = 90 - angle
                    else:
                        angle = -angle
                    if abs(angle) > 0.5 and abs(angle) < 20.0:
                        (h_c, w_c) = norm.shape[:2]
                        center = (w_c // 2, h_c // 2)
                        m_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                        norm = cv2.warpAffine(norm, m_rot, (w_c, h_c), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            except Exception as skew_err:
                logger.debug(f"Deskew skipped: {skew_err}")

            # 3. Upscale if width is smaller than optimal OCR resolution (2200px)
            h_n, w_n = norm.shape
            if w_n < 2200:
                scale = 2200.0 / w_n
                norm = cv2.resize(norm, (int(w_n * scale), int(h_n * scale)), interpolation=cv2.INTER_CUBIC)

            # 4. Subtle unsharp mask to crispen text edges without amplifying noise
            gaussian = cv2.GaussianBlur(norm, (0, 0), 2.0)
            sharpened = cv2.addWeighted(norm, 1.3, gaussian, -0.3, 0)
            return Image.fromarray(sharpened)
        except Exception as e:
            logger.warning(f"Advanced image preprocessing skipped: {e}")
            return image if isinstance(image, Image.Image) else Image.open(image)

    def extract_with_layout(self, image: Any) -> Dict[str, Any]:
        """
        Extracts both full raw text and layout-segmented regions (Header, Buyer, Invoice Meta,
        Goods Table, E-Invoice, Taxes, Remarks, Bank Details) using adaptive computer vision.
        """
        if not self.is_available():
            raise RuntimeError(
                "Tesseract OCR is not installed or not found in system PATH. "
                "Please install Tesseract OCR and set TESSERACT_CMD in .env"
            )

        import pytesseract

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        processed_img = self.preprocess_image(image)
        norm_np = np.array(processed_img)
        h, w = norm_np.shape[:2]

        layout_regions: Dict[str, str] = {}
        try:
            # Detect horizontal grid and divider lines
            thresh = cv2.adaptiveThreshold(~norm_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w / 12), 1))
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
            h_proj = np.sum(h_lines, axis=1)

            dividers = []
            for y in range(len(h_proj)):
                if h_proj[y] > (w * 0.15 * 255):
                    if not dividers or (y - dividers[-1]) > int(h * 0.015):
                        dividers.append(y)

            # Adaptive boundary determination based on relative vertical geometry
            header_divs = [d for d in dividers if 0.12 * h <= d <= 0.22 * h]
            header_y = header_divs[-1] if header_divs else int(0.18 * h)

            table_top_divs = [d for d in dividers if 0.26 * h <= d <= 0.38 * h]
            table_top_y = table_top_divs[0] if table_top_divs else int(0.32 * h)

            table_bot_divs = [d for d in dividers if 0.46 * h <= d <= 0.60 * h]
            table_bot_y = table_bot_divs[-1] if table_bot_divs else int(0.55 * h)

            einv_bot_divs = [d for d in dividers if 0.63 * h <= d <= 0.72 * h]
            einv_bot_y = einv_bot_divs[0] if einv_bot_divs else int(0.66 * h)

            bank_top_divs = [d for d in dividers if 0.80 * h <= d <= 0.92 * h]
            bank_top_y = bank_top_divs[-1] if bank_top_divs else int(0.85 * h)

            header_crop = norm_np[0:header_y, :]
            buyer_crop = norm_np[header_y:table_top_y, 0:int(w * 0.52)]
            meta_crop = norm_np[header_y:table_top_y, int(w * 0.48):w]
            goods_crop = norm_np[table_top_y:table_bot_y, :].copy()
            einv_crop = norm_np[table_bot_y:einv_bot_y, 0:int(w * 0.52)]
            tax_crop = norm_np[table_bot_y:einv_bot_y, int(w * 0.50):w]
            rem_crop = norm_np[einv_bot_y:bank_top_y, :]
            bank_crop = norm_np[bank_top_y:int(h * 0.99), 0:int(w * 0.60)]

            # Table line removal for pristine tabular text recognition without grid line noise
            g_thresh = cv2.adaptiveThreshold(~goods_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
            gh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w / 15), 1))
            gh_lines = cv2.morphologyEx(g_thresh, cv2.MORPH_OPEN, gh_kernel)
            gv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(goods_crop.shape[0] / 10)))
            gv_lines = cv2.morphologyEx(g_thresh, cv2.MORPH_OPEN, gv_kernel)
            g_table_lines = cv2.add(gh_lines, gv_lines)
            g_dilated = cv2.dilate(g_table_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            goods_no_lines = goods_crop.copy()
            goods_no_lines[g_dilated > 0] = 255

            layout_regions["header"] = pytesseract.image_to_string(header_crop, config="--psm 6").strip()
            layout_regions["buyer"] = pytesseract.image_to_string(buyer_crop, config="--psm 6").strip()
            layout_regions["invoice_meta"] = pytesseract.image_to_string(meta_crop, config="--psm 6").strip()
            layout_regions["goods_table"] = pytesseract.image_to_string(goods_no_lines, config="--psm 6").strip()
            layout_regions["e_invoice"] = pytesseract.image_to_string(einv_crop, config="--psm 6").strip()
            layout_regions["taxes"] = pytesseract.image_to_string(tax_crop, config="--psm 6").strip()
            layout_regions["remarks"] = pytesseract.image_to_string(rem_crop, config="--psm 6").strip()
            layout_regions["bank"] = pytesseract.image_to_string(bank_crop, config="--psm 6").strip()
        except Exception as seg_err:
            logger.warning(f"Layout region segmentation skipped: {seg_err}")

        # Extract full text as baseline
        full_text = pytesseract.image_to_string(processed_img, config="--psm 6").strip()
        if len(full_text) < 50:
            full_text = pytesseract.image_to_string(processed_img, config="--psm 3").strip()

        # If layout regions were successfully extracted, compose structured section text
        if layout_regions and any(len(v) > 10 for v in layout_regions.values()):
            section_blocks = []
            for sec_name, sec_content in layout_regions.items():
                if sec_content:
                    section_blocks.append(f"=== SECTION: {sec_name.upper()} ===\n{sec_content}")
            composite_text = "\n\n".join(section_blocks)
            if len(composite_text) > len(full_text) * 0.5:
                raw_text_output = f"{composite_text}\n\n=== FULL DOCUMENT OCR ===\n{full_text}"
            else:
                raw_text_output = full_text
        else:
            raw_text_output = full_text

        return {
            "raw_text": raw_text_output,
            "layout_regions": layout_regions
        }

    def extract_text(self, image: Image.Image) -> str:
        res = self.extract_with_layout(image)
        return res["raw_text"]
