"""
FiftyOne model implementation for LightOnOCR with batching support.

LightOnOCR is a vision-language model optimized for optical character recognition.
It extracts text from images using a chat-based interface.

Usage:
    import fiftyone as fo
    from lightonocr import load_model
    
    model = load_model()
    dataset.apply_model(model, label_field="ocr_text", batch_size=8)
"""

import logging
from typing import List, Union, Optional

from PIL import Image
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem

from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"




class LightOnOCRGetItem(GetItem):
    """GetItem transform for batching LightOnOCR inference.
    
    Loads images as PIL Images for processing by LightOnOCR.
    """
    
    @property
    def required_keys(self):
        """Fields required from each sample."""
        return ["filepath"]
    
    def __call__(self, sample_dict):
        """Load and return PIL Image.
        
        Args:
            sample_dict: Dictionary with 'filepath' key
            
        Returns:
            PIL.Image: RGB image loaded from filepath
        """
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        return image


class LightOnOCR(Model, SupportsGetItem, TorchModelMixin):
    """FiftyOne model for LightOnOCR text extraction with batching support.
    
    Extracts text from images using LightOnOCR's vision-language model.
    
    Args:
        model_path: HuggingFace model ID or local path 
            (default: "lightonai/LightOnOCR-2-1B")
        max_new_tokens: Maximum tokens to generate (default: 1024)
        prompt: Optional custom prompt to prepend to the image
            (default: None, uses model's default OCR behavior)
        batch_size: Batch size for inference (default: 8)
    
    Example:
        >>> model = LightOnOCR()
        >>> dataset.apply_model(model, label_field="ocr_text", batch_size=8)
        
        >>> # With custom prompt
        >>> model = LightOnOCR(prompt="Extract all text from this document:")
        >>> dataset.apply_model(model, label_field="ocr_text")
    """
    
    def __init__(
        self,
        model_path: str = "lightonai/LightOnOCR-2-1B",
        max_new_tokens: int = 4096,
        prompt: Optional[str] = None,
        batch_size: int = 8,
        **kwargs
    ):
        SupportsGetItem.__init__(self)
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self._batch_size = batch_size
        self._preprocess = False  # Preprocessing happens in GetItem
        
        # Device setup
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        logger.info(f"Loading LightOnOCR from {model_path}")
        
        self.processor = LightOnOcrProcessor.from_pretrained(model_path)
        
        # Set left padding for correct batch generation with decoder-only models
        # Right-padding causes incorrect generation as the model generates left-to-right
        self.processor.tokenizer.padding_side = "left"
        
        # Use "auto" dtype to let transformers pick the best dtype for the hardware
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto"
        )
        self.model = self.model.to(self.device).eval()
        
        # Get the actual dtype from the loaded model for input tensor conversion
        self.dtype = next(self.model.parameters()).dtype
        logger.info(f"Model loaded with dtype: {self.dtype}")
        
        logger.info(f"LightOnOCR model loaded successfully (batch_size={self._batch_size})")

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False
    
    @property
    def media_type(self):
        """The media type processed by this model."""
        return "image"
    
    @property
    def preprocess(self):
        """Whether preprocessing should be applied.
        
        For SupportsGetItem models, preprocessing is handled by the GetItem
        transform, so this should be False when using the DataLoader path.
        """
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Set preprocessing flag."""
        self._preprocess = value
    
    @property
    def has_collate_fn(self):
        """Whether this model provides a custom collate function.
        
        Returns True since we need custom collation for variable-size images.
        """
        return True
    
    @property
    def collate_fn(self):
        """Custom collate function for the DataLoader.
        
        Returns batches as lists of PIL Images without stacking,
        since LightOnOCR handles variable-size images.
        """
        @staticmethod
        def identity_collate(batch):
            """Return batch as-is (list of PIL Images)."""
            return batch
        
        return identity_collate
    
    @property
    def ragged_batches(self):
        """Whether this model supports batches with varying sizes.
        
        Returns False to enable batching with batch_size > 1.
        LightOnOCR handles variable-size PIL Images internally.
        """
        return False
    
    @property
    def transforms(self):
        """The preprocessing transforms applied to inputs.
        
        For SupportsGetItem models, preprocessing happens in the GetItem
        transform, so this returns None.
        """
        return None
    
    @property
    def batch_size(self):
        """Current batch size."""
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        """Change batch size at runtime."""
        self._batch_size = value
        logger.info(f"Batch size changed to: {value}")
    
    def get_item(self):
        """Return the GetItem transform for batching support.
        
        Returns:
            LightOnOCRGetItem: GetItem instance for loading images
        """
        return LightOnOCRGetItem()
    
    def build_get_item(self, field_mapping=None):
        """Build the GetItem transform for batching.
        
        Args:
            field_mapping: Optional field mapping dict
            
        Returns:
            LightOnOCRGetItem: GetItem instance for loading images
        """
        return LightOnOCRGetItem(field_mapping=field_mapping)
    
    def _build_conversation(self, image: Image.Image) -> list:
        """Build conversation for a single image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Conversation list in the expected format
        """
        content = [{"type": "image", "image": image}]
        
        # Add optional text prompt
        if self.prompt:
            content.append({"type": "text", "text": self.prompt})
        
        return [{"role": "user", "content": content}]
    
    def _process_single(self, image: Image.Image) -> str:
        """Process a single image through LightOnOCR.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Extracted text string
        """
        conversation = self._build_conversation(image)
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Move to device with appropriate dtype
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() 
            else v.to(self.device) 
            for k, v in inputs.items()
        }
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        # Extract generated tokens (exclude input tokens)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return output_text
    
    def _process_batch(self, images: List[Image.Image]) -> List[str]:
        """Process a batch of images through LightOnOCR.
        
        Args:
            images: List of PIL Images to process
            
        Returns:
            List of extracted text strings
        """
        # Build conversations for all images
        conversations = [self._build_conversation(img) for img in images]
        
        # Process all conversations
        batch_inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Move to device with appropriate dtype
        batch_inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() 
            else v.to(self.device) 
            for k, v in batch_inputs.items()
        }
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **batch_inputs, 
                max_new_tokens=self.max_new_tokens
            )
        
        # Decode each output
        results = []
        input_length = batch_inputs["input_ids"].shape[1]
        
        for i in range(len(images)):
            generated_ids = output_ids[i, input_length:]
            text = self.processor.decode(generated_ids, skip_special_tokens=True)
            results.append(text)
        
        return results
    
    def predict_all(self, images, preprocess=None):
        """Batch prediction for multiple images.
        
        This method enables efficient batching when using dataset.apply_model().
        
        Args:
            images: List of PIL Images (from GetItem) or numpy arrays
            preprocess: If True, convert numpy to PIL. If None, uses self.preprocess.
        
        Returns:
            List[str]: List of extracted text strings
        """
        # Use instance preprocess flag if not specified
        if preprocess is None:
            preprocess = self._preprocess
        
        # Preprocess if needed (convert numpy to PIL)
        if preprocess:
            pil_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif not isinstance(img, Image.Image):
                    raise ValueError(f"Expected PIL Image or numpy array, got {type(img)}")
                pil_images.append(img)
            images = pil_images
        else:
            # Images should already be PIL Images from GetItem
            if images and not isinstance(images[0], Image.Image):
                raise ValueError(
                    f"When preprocess=False, images must be PIL Images. "
                    f"Got {type(images[0]) if images else 'empty list'}"
                )
        
        return self._process_batch(images)
    
    def predict(self, image, sample=None):
        """Process a single image with LightOnOCR.
        
        For batch processing, use predict_all() or dataset.apply_model() which will
        automatically use batching via the GetItem interface.
        
        Args:
            image: PIL Image or numpy array to process
            sample: FiftyOne sample (optional, for compatibility)
        
        Returns:
            str: Extracted text from the image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self._process_single(image)
