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

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Adaptive image preprocessing: illumination normalization, deskew, and resolution scaling."""
        try:
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
            return image

    def extract_with_layout(self, image: Image.Image) -> Dict[str, Any]:
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
                if h_proj[y] > (w * 0.20 * 255):
                    if not dividers or (y - dividers[-1]) > int(h * 0.015):
                        dividers.append(y)

            # In standard structured GST invoices, dividers segment sections:
            # 1. Header (y=0 to ~18% h)
            # 2. Buyer (Left) & Invoice Meta (Right) (y=~16% to ~33% h)
            # 3. Goods Table (y=~31% to ~56% h)
            # 4. E-Invoice (Left) & Taxes (Right) (y=~53% to ~72% h)
            # 5. Remarks & Totals (y=~67% to ~86% h)
            # 6. Bank Details (Left) & Signatures (Right) (y=~85% to ~98% h)
            header_y = dividers[0] if dividers and dividers[0] > int(h * 0.12) else int(h * 0.18)
            meta_y2 = dividers[1] if len(dividers) > 1 and dividers[1] > int(h * 0.25) else int(h * 0.33)
            goods_y2 = dividers[2] if len(dividers) > 2 and dividers[2] > int(h * 0.50) else int(h * 0.56)

            header_crop = norm_np[0:header_y, :]
            buyer_crop = norm_np[int(h * 0.15):meta_y2, 0:int(w * 0.55)]
            meta_crop = norm_np[int(h * 0.15):meta_y2, int(w * 0.48):w]
            goods_crop = norm_np[meta_y2:goods_y2, :]
            einv_crop = norm_np[goods_y2:int(h * 0.72), 0:int(w * 0.52)]
            tax_crop = norm_np[goods_y2:int(h * 0.72), int(w * 0.58):w]
            rem_crop = norm_np[int(h * 0.67):int(h * 0.86), :]
            bank_crop = norm_np[int(h * 0.84):int(h * 0.99), 0:int(w * 0.60)]

            layout_regions["header"] = pytesseract.image_to_string(header_crop, config="--psm 6").strip()
            layout_regions["buyer"] = pytesseract.image_to_string(buyer_crop, config="--psm 6").strip()
            layout_regions["invoice_meta"] = pytesseract.image_to_string(meta_crop, config="--psm 6").strip()
            # PSM 4 preserves column alignment in tabular data
            layout_regions["goods_table"] = pytesseract.image_to_string(goods_crop, config="--psm 4").strip()
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
