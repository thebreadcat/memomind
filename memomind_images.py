"""Image intake, WebP conversion, storage management."""

import logging
import os
import uuid
from pathlib import Path

from memomind_config import CONFIG_DIR, load_config

logger = logging.getLogger("memomind.images")

IMAGE_SIZES = {
    "full": (1920, 1080),
    "thumb": (200, 200),
    "micro": (64, 64),
}

WEBP_QUALITY = {
    "full": 85,
    "thumb": 75,
    "micro": 70,
}

STORAGE_ROOT = CONFIG_DIR / "storage" / "images"


def storage_dir(entry_id: str) -> Path:
    d = STORAGE_ROOT / entry_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_image_path(entry_id: str, size_name: str) -> Path:
    return storage_dir(entry_id) / f"{size_name}.webp"


def _pillow_available():
    try:
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def _register_heif():
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        return True
    except ImportError:
        return False


def extract_text(img) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def describe_image(img) -> str:
    try:
        from memomind_models import complete, get_provider
        provider = get_provider()
        # Text-only fallback: describe from filename context not available; skip vision for now
        _ = provider
        return ""
    except Exception:
        return ""


def process_image(input_path: str, entry_id: str) -> dict:
    if not _pillow_available():
        raise RuntimeError("Pillow is required for image processing. pip install Pillow")

    from PIL import Image, ImageOps

    ext = Path(input_path).suffix.lower()
    if ext in (".heic", ".heif"):
        if not _register_heif():
            raise RuntimeError(
                "HEIC images require pillow-heif. Install with: pip install pillow-heif"
            )

    original_size = os.path.getsize(input_path)
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size
    paths = {}
    for size_name, dimensions in IMAGE_SIZES.items():
        copy = img.copy()
        copy.thumbnail(dimensions, Image.Resampling.LANCZOS)
        out_path = get_image_path(entry_id, size_name)
        copy.save(out_path, format="WEBP", quality=WEBP_QUALITY[size_name], method=6)
        paths[size_name] = str(out_path)

    try:
        os.remove(input_path)
    except OSError:
        pass

    text = extract_text(img)
    caption = describe_image(img)

    full_size = os.path.getsize(paths["full"]) if paths.get("full") else 0
    return {
        "text": text,
        "caption": caption,
        "full_path": paths["full"],
        "thumb_path": paths["thumb"],
        "micro_path": paths["micro"],
        "original_format": ext.lstrip(".") or "unknown",
        "original_size": original_size,
        "full_size": full_size,
        "width": width,
        "height": height,
    }


def get_storage_stats() -> dict:
    config = load_config()
    warning_gb = config.get("storage_warning_gb", 2.0)
    critical_gb = config.get("storage_critical_gb", 4.0)
    total_bytes = 0
    image_count = 0
    if STORAGE_ROOT.exists():
        for p in STORAGE_ROOT.rglob("*.webp"):
            total_bytes += p.stat().st_size
            if p.name == "full.webp":
                image_count += 1
    total_gb = total_bytes / (1024**3)
    level = "ok"
    if total_gb >= critical_gb:
        level = "critical"
    elif total_gb >= warning_gb:
        level = "warning"
    return {
        "total_bytes": total_bytes,
        "total_gb": round(total_gb, 3),
        "image_count": image_count,
        "warning_gb": warning_gb,
        "critical_gb": critical_gb,
        "level": level,
    }
