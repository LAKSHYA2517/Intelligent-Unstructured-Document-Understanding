import logging
from typing import Optional
from image_processor import ImageProcessor

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_DOCUMENT = 20  # Hard cap: don't use more than this per doc


class ImageCaptionPipeline:
    """
    Controlled image captioning with:
    1. Pre-filtering (skip tiny/decorative images)
    2. Caching (never re-caption the same image)
    3. Hard cap per document (protect daily quota)
    4. Graceful degradation (missing caption != failure)
    """

    def __init__(
        self,
        managed_vision_client,
        image_processor: ImageProcessor,
        vision_model: str = "llama-4-scout-17b-16e-instruct",
        max_images_per_doc: int = MAX_IMAGES_PER_DOCUMENT
    ):
        self.client          = managed_vision_client
        self.processor       = image_processor
        self.model           = vision_model
        self.max_per_doc     = max_images_per_doc

    def caption_document_images(
        self,
        image_elements: list,
        doc_id: str
    ) -> dict[str, str]:
        """
        Caption all images in a document.
        Returns: {element_id: caption_text}
        """
        captions: dict[str, str] = {}
        captioned_count = 0

        for element in image_elements:
            if captioned_count >= self.max_per_doc:
                captions[element.element_id] = (
                    "[Image: document image limit reached, "
                    "caption not generated]"
                )
                continue

            import base64
            try:
                image_bytes = base64.b64decode(element.content)
            except Exception:
                captions[element.element_id] = "[Image: could not decode]"
                continue

            if not self.processor.should_caption(image_bytes):
                captions[element.element_id] = (
                    "[Image: decorative or too small to caption]"
                )
                continue

            cached = self.processor.get_cached_caption(image_bytes)
            if cached:
                captions[element.element_id] = cached
                logger.debug(
                    f"Using cached caption for element {element.element_id}"
                )
                continue

            b64_compressed = self.processor.prepare_image_for_api(image_bytes)
            if b64_compressed is None:
                captions[element.element_id] = (
                    "[Image: compression failed, caption not generated]"
                )
                continue

            caption = self._caption_single_image(b64_compressed, element)
            captions[element.element_id] = caption

            self.processor.save_caption_cache(image_bytes, caption)
            captioned_count += 1

        logger.info(
            f"Document {doc_id}: Captioned {captioned_count}/{len(image_elements)} "
            f"images ({len(image_elements) - captioned_count} skipped)"
        )

        return captions

    def _caption_single_image(
        self,
        b64_image: str,
        element
    ) -> str:
        CAPTION_PROMPT = (
            "Analyze this document image. In 2-3 sentences, describe: "
            "1) What type of visual is this (chart, table, diagram, photo)? "
            "2) What specific data or information does it contain? "
            "3) What is the key takeaway or finding shown? "
            "Focus on factual content, not visual style."
        )

        try:
            response = self.client.call(
                self.client.client.chat.completions.create,
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        }
                    ]
                }],
                temperature=0.0,
                max_tokens=200
            )

            return (
                f"[Image Caption: "
                f"{response.choices[0].message.content.strip()}]"
            )

        except Exception as e:
            logger.error(f"Vision captioning failed: {e}")
            return "[Image: captioning failed, content may be relevant]"
