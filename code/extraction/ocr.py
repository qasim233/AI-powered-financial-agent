"""
OCR Processor — EasyOCR wrapper for image text extraction.

Per SOLUTION.md §5: EasyOCR for all images (no direct multimodal LLM calls).
OCR text is fed into the same extraction prompt used for messages.

All 16 images are financial documents (1 payslip, 15 invoices/bills/receipts)
in English, with some Indonesian text.  The key information is numeric amounts
and labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    EasyOCR wrapper with lazy initialisation.

    The EasyOCR reader downloads models on first use; lazy init avoids this
    cost if no images need processing and keeps import time fast.
    """

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False) -> None:
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._reader = None  # lazy

    @property
    def reader(self):
        """Lazy-initialise the EasyOCR reader on first use."""
        if self._reader is None:
            try:
                import easyocr

                self._reader = easyocr.Reader(
                    self._languages, gpu=self._gpu, verbose=False
                )
                logger.info(
                    "EasyOCR reader initialised: languages=%s gpu=%s",
                    self._languages,
                    self._gpu,
                )
            except ImportError:
                logger.error(
                    "easyocr is not installed. Run: pip install easyocr"
                )
                raise
        return self._reader

    def extract_text(self, image_path: str) -> str:
        """
        Run OCR on an image and return the extracted text.

        Parameters
        ----------
        image_path:
            Absolute or relative path to a PNG image file.

        Returns
        -------
        Extracted text with lines joined by newlines, or empty string
        if the file does not exist or OCR fails.
        """
        path = Path(image_path)
        if not path.exists():
            logger.warning("Image file not found: %s", image_path)
            return ""

        try:
            # detail=1 returns (bbox, text, confidence) tuples
            # We keep the full detail to log confidence but extract just text
            results = self.reader.readtext(str(path), detail=1)

            if not results:
                logger.warning("OCR returned no text for %s", image_path)
                return ""

            # Join text segments in reading order (EasyOCR returns top-to-bottom,
            # left-to-right).  Separate segments with newlines to preserve
            # tabular structure common in financial documents.
            lines = []
            for _bbox, text, conf in results:
                lines.append(text)
                if conf < 0.3:
                    logger.debug(
                        "Low OCR confidence (%.2f) for segment in %s: '%s'",
                        conf,
                        path.name,
                        text[:50],
                    )

            extracted = "\n".join(lines)
            logger.info(
                "OCR extracted %d segments (%d chars) from %s",
                len(results),
                len(extracted),
                path.name,
            )
            return extracted

        except Exception:
            logger.exception("OCR failed for %s", image_path)
            return ""
