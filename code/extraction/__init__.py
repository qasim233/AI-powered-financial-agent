"""Extraction package — EasyOCR + LLM structured claim extraction."""

from .claim import Claim

def get_ocr_processor(*args, **kwargs):
    from .ocr import OCRProcessor
    return OCRProcessor(*args, **kwargs)

def get_llm_extractor(*args, **kwargs):
    from .llm_extractor import LLMExtractor
    return LLMExtractor(*args, **kwargs)

__all__ = ["Claim", "get_ocr_processor", "get_llm_extractor"]
