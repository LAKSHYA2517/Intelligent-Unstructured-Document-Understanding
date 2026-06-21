import base64
import hashlib
import io
import os
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# The real limits for free tier vision APIs
# Groq vision: recommends <1MB per image, drops connection on larger
# Gemini Flash: 4MB limit but drops TCP on repeated large payloads
MAX_IMAGE_BYTES_FOR_API = 100_000   # 100KB hard limit
TARGET_IMAGE_BYTES      = 50_000    # 50KB target (extremely safe)
MAX_DIMENSION           = 300       # Further reduced to guarantee small payloads


class ImageProcessor:
    """
    Responsible for:
    1. Compressing images to a size that won't cause TCP connection drops
    2. Caching captions so we never re-caption the same image
    3. Skipping images that are clearly not worth captioning
    """

    def __init__(self, cache_dir: str = ".image_caption_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def prepare_image_for_api(
        self, 
        image_bytes: bytes,
        aggressive: bool = False  # Use aggressive compression if normal fails
    ) -> Optional[str]:
        """
        Compress an image until it is under the size limit.
        Returns base64 string, or None if image is too small to be worth sending.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB (removes alpha channel, reduces size)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Check if image is large enough to be meaningful
            w, h = img.size
            if w < 50 or h < 50:
                logger.debug(f"Skipping tiny image ({w}x{h})")
                return None

            max_dim = MAX_DIMENSION if not aggressive else MAX_DIMENSION // 2

            # Try progressively lower quality until under size limit
            quality_levels = [70, 50, 35, 20] if not aggressive else [30, 15]

            for quality in quality_levels:

                # Resize if needed
                scale = min(max_dim / w, max_dim / h, 1.0)
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    resized = img

                # Compress to JPEG
                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=quality, optimize=True)
                compressed_bytes = buffer.getvalue()

                if len(compressed_bytes) <= TARGET_IMAGE_BYTES:
                    b64 = base64.b64encode(compressed_bytes).decode("utf-8")
                    logger.debug(
                        f"Image compressed: {len(image_bytes):,} → "
                        f"{len(compressed_bytes):,} bytes "
                        f"(quality={quality}, scale={scale:.2f})"
                    )
                    return b64

                # Make image smaller for next attempt
                max_dim = int(max_dim * 0.75)
                w, h = resized.size

            # If still over limit, force to thumbnail
            img.thumbnail((128, 128), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=15)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.warning(
                "Image required extreme compression to fit under size limit"
            )
            return b64

        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            return None

    def get_cache_key(self, image_bytes: bytes) -> str:
        """SHA256 hash of image bytes as cache key."""
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def get_cached_caption(self, image_bytes: bytes) -> Optional[str]:
        """Return cached caption if available."""
        key = self.get_cache_key(image_bytes)
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def save_caption_cache(self, image_bytes: bytes, caption: str) -> None:
        """Save caption to disk cache."""
        key = self.get_cache_key(image_bytes)
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(caption, encoding="utf-8")

    def should_caption(self, image_bytes: bytes) -> bool:
        """
        Decide if an image is worth sending to the vision API.
        Saves vision API quota.
        """
        if len(image_bytes) < 5_000:   # < 5KB
            return False

        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 20:  # Very elongated
                return False
            if w < 80 or h < 80:
                return False
        except Exception:
            pass

        return True
