import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from models.utils import unwrap_model_for_generation
from modeling_base import create_reference_model
from grpo_config import GRPOConfig
from utils import generate_model_card, get_comet_experiment_url

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_vllm_available():
#     from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# class RepeatRandomSampler(Sampler):
#     """
#     Sampler that repeats the indices of a dataset N times.

#     Args:
#         data_source (`Sized`):
#             Dataset to sample from.
#         repeat_count (`int`):
#             Number of times to repeat each index.

#     Example:
#     ```python
#     >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
#     >>> list(sampler)
#     [2, 2, 0, 0, 3, 3, 1, 1]
#     ```
#     """

#     def __init__(self, data_source: Sized, repeat_count: int):
#         self.data_source = data_source
#         self.repeat_count = repeat_count
#         self.num_samples = len(data_source)

#     def __iter__(self):
#         indexes = [idx for idx in torch.randperm(self.num_samples).tolist() for _ in range(self.repeat_count)]
#         return iter(indexes)

#     def __len__(self):
#         return self.num_samples * self.repeat_count


class GRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin, list[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None        

        # Processing class for VL models
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, padding_side="left")

        # # Preprocess datasets
        # if train_dataset is not None:
        #     train_dataset = train_dataset.map(
        #         lambda example: self.process_row(
        #             #remove_columns=["prompt", "chosen", "rejected"], TODO
        #             example,
        #             processing_class,
        #             32, # TODO:  self.max_prompt_length, # TODO: self.max_completion_length,
        #             32,
        #             add_special_tokens=False
        #         ),
        #         batched=False,
        #     )
        # if eval_dataset is not None:
        #     if isinstance(eval_dataset, dict):
        #         for key in eval_dataset:
        #             eval_dataset[key] = eval_dataset[key].map(
        #                 lambda example: self.process_row(
        #                     example,
        #                     processing_class,
        #                     self.max_prompt_length,
        #                     self.max_completion_length,
        #                     add_special_tokens=False
        #                 ),
        #                 batched=False,
        #             )
        #     else:
        #         eval_dataset = eval_dataset.map(
        #             lambda example: self.process_row(
        #                 example,
        #                 processing_class,
        #                 self.max_prompt_length,
        #                 self.max_completion_length,
        #                 add_special_tokens=False
        #             ),
        #             batched=False,
        #         )



        def data_collator(features):
        # print(len([f["pixel_values"] for f in features][0]))

        # pixel_values = torch.stack([f["pixel_values"] for f in features])
        # prompt_input_ids = [torch.tensor(f["prompt_input_ids"]) for f in features]
        # prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
        #     prompt_input_ids,
        #     batch_first=True,
        #     padding_value=processing_class.tokenizer.pad_token_id
        # )
        # batch = {
        #     "pixel_values": pixel_values,
        #     "prompt_input_ids": prompt_input_ids,
        #     "prompt": [f["prompt"] for f in features],
        # }
        # if "pixel_attention_mask" in features[0]:
        #     batch["pixel_attention_mask"] = torch.stack([f["pixel_attention_mask"] for f in features])
        # if "image_sizes" in features[0]:
        #     batch["image_sizes"] = torch.stack([f["image_sizes"] for f in features])
            return features


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoProcessor.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

  
    @staticmethod
    def process_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens
    ):
        processor = processing_class
        processed_features = processor(images=features["image"], text="<image>" + features["question"], add_special_tokens=False)
        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"][0] #TODO does  GRPO expect batch or single

        if add_special_tokens:
            if processor.tokenizer.bos_token_id is not None:
                prompt_input_ids = [processor.tokenizer.bos_token_id] + prompt_input_ids
            if processor.tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [processor.tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "prompt": features["question"],
        }

        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]

        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        prompt_input_ids = inputs["prompt_input_ids"]
        pixel_values = inputs["pixel_values"]
        prompts = inputs["prompt"]

        # Prepare generation inputs
        generation_inputs = {
            "input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "attention_mask": (prompt_input_ids != self.processing_class.tokenizer.pad_token_id).int(),
        }
        if "pixel_attention_mask" in inputs:
            generation_inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
        if "image_sizes" in inputs:
            generation_inputs["image_sizes"] = inputs["image_sizes"]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **generation_inputs,
                generation_config=self.generation_config
            )

        # Get the per-token log probabilities for the completions 
        # (now handles vision-language model inputs)
        def get_per_token_logps(model, input_ids, pixel_values, attention_mask=None, **kwargs):
            # Forward pass with both text and image inputs
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **kwargs
            )
            logits = outputs.logits  # (B, L, V)
            logits = logits[:, :-1, :]  # Shift to align with input_ids[1:]
            input_ids = input_ids[:, 1:]  # Remove first token
            
            # Calculate per-token log probabilities
            log_probs = logits.log_softmax(dim=-1)
            per_token_logps = torch.gather(
                log_probs, 
                dim=-1, 
                index=input_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            return per_token_logps

        # Calculate log probabilities for current policy
        per_token_logps = get_per_token_logps(
            model,
            prompt_completion_ids,
            pixel_values=pixel_values.repeat_interleave(self.num_generations, dim=0),
            attention_mask=generation_inputs.get("attention_mask")
        )

        # Calculate reference model log probabilities
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    pixel_values=pixel_values.repeat_interleave(self.num_generations, dim=0),
                    attention_mask=generation_inputs.get("attention_mask")
                )
            else:
                # Handle PEFT case by disabling adapters
                with model.disable_adapter():
                    ref_per_token_logps = get_per_token_logps(
                        model,
                        prompt_completion_ids,
                        pixel_values=pixel_values.repeat_interleave(self.num_generations, dim=0),
                        attention_mask=generation_inputs.get("attention_mask")
                    )

        # Remove prompt parts from log probabilities
        prompt_length = prompt_input_ids.size(1)
        per_token_logps = per_token_logps[:, prompt_length-1:]
        ref_per_token_logps = ref_per_token_logps[:, prompt_length-1:]

        # Calculate KL divergence per token
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                      (ref_per_token_logps - per_token_logps) - 1

        # Create completion mask (ignore padding and post-EOS tokens)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_positions = is_eos.int().argmax(dim=1, keepdim=True)
        sequence_lengths = torch.where(
            is_eos.any(dim=1),
            eos_positions.squeeze(-1) + 1,
            torch.tensor(completion_ids.size(1), device=device)
        )
        completion_mask = torch.arange(
            completion_ids.size(1), device=device
        )[None, :] < sequence_lengths[:, None]

        # Decode completions for reward calculation
        completions = self.processing_class.tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True
        )

        # Prepare inputs for reward functions
        batch_size = len(prompts)
        expanded_images = pixel_values.repeat_interleave(self.num_generations, dim=0)
        expanded_prompts = [p for p in prompts for _ in range(self.num_generations)]

        # Calculate rewards using all reward functions
        rewards_per_func = torch.zeros(len(expanded_prompts), len(self.reward_funcs), device=device)
        
        for i, (reward_func, reward_processor) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                # Handle vision-language reward models
                reward_inputs = reward_processor(
                    text=[p + c for p, c in zip(expanded_prompts, completions)],
                    images=expanded_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length + self.max_completion_length,
                    add_special_tokens=False
                ).to(device)

                with torch.inference_mode():
                    rewards = reward_func(**reward_inputs).logits.squeeze(-1)
            else:
                # Custom reward function handling
                rewards = torch.tensor(
                    reward_func(
                        prompts=expanded_prompts,
                        completions=completions,
                        images=expanded_images
                    ),
                    device=device
                )
            
            rewards_per_func[:, i] = rewards

        # Normalize rewards across groups
        total_rewards = rewards_per_func.sum(dim=1)
        group_rewards = total_rewards.view(batch_size, self.num_generations)
        mean_rewards = group_rewards.mean(dim=1, keepdim=True)
        std_rewards = group_rewards.std(dim=1, keepdim=True) + 1e-8
        normalized_rewards = (group_rewards - mean_rewards) / std_rewards
        advantages = normalized_rewards.view(-1, 1)

        # Calculate final loss with KL penalty
        policy_advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        kl_penalized_loss = (policy_advantages - self.beta * per_token_kl) * completion_mask
        normalized_loss = kl_penalized_loss.sum(dim=1) / completion_mask.sum(dim=1)
        final_loss = -normalized_loss.mean()

        # Log metrics
        self._metrics["reward"].append(total_rewards.mean().item())
        self._metrics["kl"].append(per_token_kl.mean().item())
        self._metrics["completion_length"].append(completion_mask.sum(dim=1).float().mean().item())

        return final_loss

    