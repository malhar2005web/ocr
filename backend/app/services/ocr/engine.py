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
        """Preprocesses image using OpenCV to improve OCR accuracy."""
        try:
            # Convert PIL to OpenCV format (BGR)
            np_img = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

            # Resize if image resolution is too low
            height, width = gray.shape
            if width < 1000:
                scale = 1000 / width
                gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

            # Denoise and adaptive thresholding
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresholded = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            return Image.fromarray(thresholded)
        except Exception as e:
            logger.warning(f"OpenCV image preprocessing skipped: {e}")
            return image

    def extract_text(self, image: Image.Image) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Tesseract OCR is not installed or not found in system PATH. "
                "Please install Tesseract OCR and set TESSERACT_CMD in .env"
            )
        
        import pytesseract
        processed = self.preprocess_image(image)
        # PSM 3: Fully automatic page segmentation without OSD
        text = pytesseract.image_to_string(processed, config="--psm 3")
        return text.strip()
