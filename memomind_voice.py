"""Whisper integration, browser fallback."""

import logging
import os
import platform
import tempfile
from pathlib import Path

from memomind_config import load_config

logger = logging.getLogger("memomind.voice")

_whisper_model = None
_whisper_available = None

WHISPER_MODELS = {
    "pi": "tiny",
    "mac": "base",
    "desktop": "small",
}


def detect_platform_model() -> str:
    config = load_config()
    if config.get("whisper_model"):
        return config["whisper_model"]
    machine = platform.machine().lower()
    system = platform.system().lower()
    if "arm" in machine or "aarch" in machine:
        return WHISPER_MODELS["pi"]
    if system == "darwin":
        return WHISPER_MODELS["mac"]
    return WHISPER_MODELS["desktop"]


def is_whisper_available() -> bool:
    global _whisper_available
    if _whisper_available is not None:
        return _whisper_available
    try:
        import whisper  # noqa: F401
        _whisper_available = True
    except ImportError:
        _whisper_available = False
    return _whisper_available


def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    if not is_whisper_available():
        return None
    import whisper
    model_name = detect_platform_model()
    logger.info("Loading Whisper model: %s", model_name)
    _whisper_model = whisper.load_model(model_name)
    return _whisper_model


def transcribe_audio(audio_data: bytes, filename: str = "audio.webm") -> dict:
    """Transcribe audio bytes. Returns {text, source}."""
    if not audio_data:
        return {"text": "", "source": "none", "error": "No audio data"}

    if is_whisper_available():
        try:
            model = get_whisper_model()
            suffix = Path(filename).suffix or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            try:
                result = model.transcribe(temp_path)
                text = result.get("text", "").strip()
                return {"text": text, "source": "whisper"}
            finally:
                os.unlink(temp_path)
        except Exception as e:
            logger.warning("Whisper transcription failed: %s", e)
            return {
                "text": "",
                "source": "browser_fallback",
                "error": str(e),
                "message": "Whisper failed — use browser dictation",
            }

    return {
        "text": "",
        "source": "browser_fallback",
        "message": "Whisper not installed — use browser dictation",
    }


def status() -> dict:
    return {
        "whisper_available": is_whisper_available(),
        "whisper_model": detect_platform_model() if is_whisper_available() else None,
        "browser_fallback": True,
    }
