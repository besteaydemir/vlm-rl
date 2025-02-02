from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Adds vision-language model support parameters.
    """
    model_init_kwargs: Optional[dict] = field(
    default=None,
    metadata={
        "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
        "argument of the `GRPOTrainer` is provided as a string."
    },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={"help": "Number of generations to sample."},
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )
    
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