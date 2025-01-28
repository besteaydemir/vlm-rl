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

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, AutoProcessor]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args initialization and model setup remains mostly the same
        # ...

        # Processing class for VL models
        if processing_class is None:
            try:
                processing_class = AutoProcessor.from_pretrained(model_id, padding_side="left")
            except Exception:
                processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        # Preprocess datasets
        if train_dataset is not None:
            train_dataset = train_dataset.map(
                lambda example: self.process_row(
                    example,
                    processing_class,
                    256, # TODO:  self.max_prompt_length, # TODO: self.max_completion_length,
                    32,
                    add_special_tokens=True
                ),
                batched=False,
            )
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                for key in eval_dataset:
                    eval_dataset[key] = eval_dataset[key].map(
                        lambda example: self.process_row(
                            example,
                            processing_class,
                            self.max_prompt_length,
                            self.max_completion_length,
                            add_special_tokens=True
                        ),
                        batched=False,
                    )
            else:
                eval_dataset = eval_dataset.map(
                    lambda example: self.process_row(
                        example,
                        processing_class,
                        self.max_prompt_length,
                        self.max_completion_length,
                        add_special_tokens=True
                    ),
                    batched=False,
                )

        # Data collator
        def data_collator(features):
            pixel_values = torch.stack([f["pixel_values"] for f in features])
            prompt_input_ids = [torch.tensor(f["prompt_input_ids"]) for f in features]
            prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
                prompt_input_ids,
                batch_first=True,
                padding_value=processing_class.tokenizer.pad_token_id
            )
            batch = {
                "pixel_values": pixel_values,
                "prompt_input_ids": prompt_input_ids,
                "prompt": [f["prompt"] for f in features],
            }
            if "pixel_attention_mask" in features[0]:
                batch["pixel_attention_mask"] = torch.stack([f["pixel_attention_mask"] for f in features])
            if "image_sizes" in features[0]:
                batch["image_sizes"] = torch.stack([f["image_sizes"] for f in features])
            return batch

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

    @staticmethod
    def process_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens
    ):
        processor = processing_class
        processed_features = processor(images=features["images"], text=features["prompt"], add_special_tokens=False)
        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"][0]

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
            "prompt": features["prompt"],
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