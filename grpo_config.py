from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Adds vision-language model support parameters.
    """
    # Existing parameters remain unchanged
    
    # New parameters for VL support
    image_processor_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for image preprocessing when using vision-language models"
        },
    )
    max_image_size: Optional[int] = field(
        default=224,
        metadata={
            "help": "Maximum size for resizing images (width/height)"
        },
    )
    keep_original_images: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep original images in dataset features for reward calculation"
        },
    )

    # Modified parameter metadata
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum token length for text prompts. For VL models, includes both text and image tokens."
        },
    )