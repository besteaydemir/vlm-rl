import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch.utils.data import Sampler
from torch import nn
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
from extras import profiling_context, profiling_decorator
from models.utils import unwrap_model_for_generation
from modeling_base import create_reference_model
from grpo_config import GRPOConfig
from utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax, print_prompt_completions_sample
from import_utils import is_rich_available, is_vllm_available


from PIL import Image

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_vllm_available():
#     from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count



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
            print("getting the peft config")
            model = get_peft_model(model, peft_config)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Trainable: {name}")

        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)      

        # Processing class for VL models
        # if processing_class is None:
        #     processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                raise NotImplementedError("AutoModelForSequenceClassification from string reward function is not implemented yet.")
                # reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                #     reward_func, num_labels=1, **model_init_kwargs
                # )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        # for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
        #     if isinstance(reward_func, PreTrainedModel):
        #         if reward_processing_class is None:
        #             reward_processing_class = AutoProcessor.from_pretrained(reward_func.config._name_or_path)
        #         if reward_processing_class.pad_token_id is None:
        #             reward_processing_class.pad_token = reward_processing_class.eos_token
        #         # The reward model computes the reward for the latest non-padded token in the input sequence.
        #         # So it's important to set the pad token ID to the padding token ID of the processing class.
        #         reward_func.config.pad_token_id = reward_processing_class.pad_token_id
        #         reward_processing_classes[i] = reward_processing_class
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                # if reward_processing_class.pad_token_id is None:
                #     reward_processing_class.pad_token = reward_processing_class.eos_token TODO: My reward function has a get score function
                # # The reward model computes the reward for the latest non-padded token in the input sequence.
                # # So it's important to set the pad token ID to the padding token ID of the processing class.
                # reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # for i, reward_func in enumerate(self.reward_funcs):
        #     if isinstance(reward_func, PreTrainedModel):
        #         self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
            # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

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

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                device_type = PartialState().default_device.type
                device_module = getattr(torch, device_type)
                if vllm_device == "auto":
                    if device_module.device_count() == 1:
                        vllm_device = f"{device_type}:0"  # particular case when training with onyl 1 device: share it
                    else:
                        vllm_device = f"{device_type}:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == f"{device_type}"
                    and int(vllm_device.split(":")[1]) >= device_module.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {device_module.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"{device_type}:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )

                # For Ascend NPU (torch-npu), collective communication requires the establishment of a communication
                # group, and different processes must hold the same group number. However, multiple process groups will
                # be created internally within vLLM. This will cause the group id of the communication group on rank 0
                # to be different from that of other ranks, causing backward to hang on because the communication
                # domain cannot be established. So we need to patch it to make sure the group id of different ranks in
                # the training phase are the same.
                @contextlib.contextmanager
                def new_group_context():
                    new_group = torch.distributed.new_group
                    try:
                        torch.distributed.new_group = functools.partial(new_group, use_local_synchronization=True)
                        torch.npu.mem_get_info = functools.partial(torch.npu.mem_get_info, device=vllm_device)
                        yield
                    finally:
                        torch.distributed.new_group = new_group

                new_group_patch = new_group_context() if device_type == "npu" else contextlib.nullcontext()
                with world_size_patch, profiling_patch, new_group_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                        max_model_len=self.args.vllm_max_model_len,
                    )

                # Guided decoding, if enabled
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex)
                else:
                    guided_decoding = None

                # Sampling parameters
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.tokenizer.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)



    def _set_signature_columns_if_needed(self): #TODO see this after the dataset
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # We need a custom sampler that samples the same prompt multiple times
    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )


    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            print("model is peft")
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    # Get the per-token log probabilities for the completions 
    # (now handles vision-language model inputs)
    def _get_per_token_logps(self, model, input_ids, pixel_values, logits_to_keep, attention_mask=None, pixel_attention_mask=None, **kwargs):
        # Forward pass with both text and image inputs
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        outputs = model( #TODO bring back the normal version if passed to a text only model
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            #num_logits_to_keep=logits_to_keep + 1,
            **kwargs
        )
        logits = outputs.logits  # (B, L, V)
        logits = logits[:, :-1, :]  #exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, -logits_to_keep:]  # Remove first token
        

        logits = logits[:, -logits_to_keep:]
        # # Calculate per-token log probabilities
        # log_probs = logits.log_softmax(dim=-1)
        # per_token_logps = torch.gather(
        #     log_probs, 
        #     dim=-1, 
        #     index=input_ids.unsqueeze(-1)
        # ).squeeze(-1)
        
        # return pere_token_logps
        return selective_log_softmax(logits, input_ids)
  

    # Used inside training_step in the Trainer class
    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["question"] for x in inputs]  # list for the whole inputs
        images = [x["image"] for x in inputs]
        message_list = [[{"role": "user",
                            "content": [
                                            {"type": "image"},
                                            {"type": "text", "text": prompt} 
                                        ]}]
                                        for prompt in prompts
                                    ]
        prompts_text = [self.processing_class.apply_chat_template(message, add_generation_prompt=True, continue_final_message=False, tokenize=False) for message in message_list]
        prompt_inputs = self.processing_class(
            text = prompts_text, images = images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values, pixel_attention_mask =  prompt_inputs["pixel_values"], prompt_inputs["pixel_attention_mask"] 


        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    input_ids =prompt_ids, pixel_values=pixel_values, attention_mask=prompt_mask, pixel_attention_mask=pixel_attention_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, pixel_values, logits_to_keep, attention_mask, pixel_attention_mask
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, pixel_values, logits_to_keep, attention_mask, pixel_attention_mask
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    elif True: #TODO change this
                        #print("my verison")
                        def chat_version(prompt, completion):
                            chat = [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": completion}]
                            return chat
                        
                        texts = [chat_version(p,c) for p, c in zip(prompts, completions)]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    # reward_inputs = reward_processing_class(
                    #     texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    # )
                    # reward_inputs = super()._prepare_inputs(reward_inputs)
                    # with torch.inference_mode():
                    #     rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                    images_list = [input_one["image"] for input_one in inputs]
                    image_paths = []
                    saving_path = "image.jpg"
                    for image_ in images_list:
                        image_.save(saving_path)
                        image_paths.append([saving_path])
                    
                    #print(image_paths)
                    #print(images_list)
                    with torch.inference_mode():

                        # print reward functions device
                        #print("reward function device", reward_func.device)
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            rewards_per_func[:, i] = torch.FloatTensor(reward_func.get_scores(texts, image_paths, hd_num=2)).to('cuda') # Shape (B*G,)
                    #print("reward per func", rewards_per_func)


                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                # if is_rich_available():
                #     print_prompt_completions_sample(
                #         prompts_to_log,
                #         completions_to_log,
                #         rewards_to_log,
                #         self.state.global_step,
                #     )
                # if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                #     import pandas as pd

                #     # For logging
                #     table = {
                #         "step": [str(self.state.global_step)] * len(rewards),
                #         "prompt": prompts_to_log,
                #         "completion": completions_to_log,
                #         "reward": rewards.tolist(),
                #     }
                #     df = pd.DataFrame(table)
                #     wandb.log({"completions": wandb.Table(dataframe=df)})
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd
                    if not hasattr(self, "wandb_table"):
                        self.wandb_table = wandb.Table(columns=["step", "prompt", "completion", "reward"])

                    # Append new data
                    for p, c, r in zip(prompts_to_log, completions_to_log, rewards_to_log):
                        self.wandb_table.add_data(str(self.state.global_step), p, c, r)

                    # Log without overwriting
                    wandb.log({"completions": self.wandb_table})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        pixel_values, pixel_attention_mask = inputs["pixel_values"], inputs["pixel_attention_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, pixel_values, logits_to_keep, attention_mask, pixel_attention_mask)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    # def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
    #     print("prepare")
    #     print("inputs here")
    #     print(inputs)
    #     assert 1 == 2
    #     device = self.accelerator.device
    #     prompts = [x["question"] for x in inputs]  # list for the whole inputs
    #     images = [x["image"] for x in inputs]
    #     message_list = [     [{"role": "user", "content": [
    #                                         {"type": "image"},
    #                                         {"type": "text", "text": prompt}
    #                                     ]}]
    #                                     for prompt in prompts
    #                                 ]
    #     prompts_text = [self.processing_class.apply_chat_template(message, add_generation_prompt=True) for message in message_list]
    #     prompt_inputs = self.processing_class(
    #         text = prompts_text, images = images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    #     )
    #     prompt_inputs = super()._prepare_inputs(prompt_inputs)
    #     prompt_ids, prompt_mask, pixel_values = prompt_inputs["input_ids"], prompt_inputs["attention_mask"], prompt_inputs["pixel_values"]
    #     pixel_attention_mask = prompt_inputs["pixel_attention_mask"] 

    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]

    #     # Generate completions using either vLLM or regular generation
    #     if self.args.use_vllm:
    #         # First, have main process load weights if needed
    #         if self.state.global_step != self._last_loaded_step:
    #             with unwrap_model_for_generation(
    #                 self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
    #             ) as unwrapped_model:
    #                 if is_compiled_module(unwrapped_model): 
    #                     state_dict = unwrapped_model._orig_mod.state_dict()
    #                 else:
    #                     state_dict = unwrapped_model.state_dict()
    #             if self.accelerator.is_main_process:
    #                 llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
    #                 llm_model.load_weights(state_dict.items())
    #             self._last_loaded_step = self.state.global_step

    #         # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
    #         all_prompts_text = gather_object(prompts_text)
    #         if self.accelerator.is_main_process:
    #             outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
    #             completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
    #         else:
    #             completion_ids = [None] * len(all_prompts_text)
    #         # Broadcast the completions from the main process to all processes, ensuring each process receives its
    #         # corresponding slice.
    #         completion_ids = broadcast_object_list(completion_ids, from_process=0)
    #         process_slice = slice(
    #             self.accelerator.process_index * len(prompts),
    #             (self.accelerator.process_index + 1) * len(prompts),
    #         )
    #         completion_ids = completion_ids[process_slice]

    #         # Pad the completions, and concatenate them with the prompts
    #         completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    #         completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
    #         prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    #     else:
    #         # Regular generation path
    #         with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
    #             prompt_completion_ids = unwrapped_model.generate(
    #                 input_ids =prompt_ids, pixel_values=pixel_values, attention_mask=prompt_mask, pixel_attention_mask=pixel_attention_mask, generation_config=self.generation_config
    #             )

    #         # Compute prompt length and extract completion ids
    #         prompt_length = prompt_ids.size(1)
    #         prompt_ids = prompt_completion_ids[:, :prompt_length]
    #         completion_ids = prompt_completion_ids[:, prompt_length:]

    #     # Mask everything after the first EOS token
    #     is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    #     # Concatenate prompt_mask with completion_mask for logit computation
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

    #     logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    #     with torch.inference_mode(): 
    #         if self.ref_model is not None:
    #             ref_per_token_logps = self._get_per_token_logps(
    #                 self.ref_model, prompt_completion_ids, pixel_values, logits_to_keep, attention_mask
    #             )
    #         else:
    #             with self.accelerator.unwrap_model(self.model).disable_adapter():
    #                 ref_per_token_logps = self._get_per_token_logps(
    #                     self.model, prompt_completion_ids, pixel_values, logits_to_keep, attention_mask
    #                 )

    #     # Decode the generated completions
    #     completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    #     print(completions_text)
    #     if is_conversational(inputs[0]):
    #         #print("is conversational")
    #         completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
    #         #print(completions)
    #     else:
    #         #print("not conversational")
    #         completions = completions_text
    #         #print(completions)

    #     ##print("completions", completions)

    #     rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    #     for i, (reward_func, reward_processing_class) in enumerate(
    #         zip(self.reward_funcs, self.reward_processing_classes)
    #     ):
    #         if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
    #             #print("here")
    #             #print(inputs[0].keys())
    #             #print("here")
    #             #print(prompts)
    #             if is_conversational(inputs[0]):
    #                 #print("is conversational")
    #                 messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
    #                 #print(messages)
    #                 texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
    #                 #print("texts", texts)
    #             elif True: #TODO change this
    #                 #print("my verison")
    #                 def chat_version(prompt, completion):
    #                     chat = [
    #                         {"role": "user", "content": prompt},
    #                         {"role": "assistant", "content": completion}]
    #                     return chat
                    
    #                 texts = [chat_version(p,c) for p, c in zip(prompts, completions)]
    #                 #print(texts)

    #             else:
    #                 #print("not conversational")
    #                 texts = [p + c for p, c in zip(prompts, completions)]
    #                 #print("texts", texts)

    #             # reward_inputs = reward_processing_class(
    #             #     texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
    #             # )
    #             # #print("reward_inputs", reward_inputs)
    #             # reward_inputs = super()._prepare_inputs(reward_inputs)
    #             # #print("reward prepared", reward_inputs)
    #             # with torch.inference_mode():
    #             #     rewards_per_func[:, i]  = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)

    #             #image_paths = [input_one["image_path"] for input_one in inputs]
    #             images_list = [input_one["image"] for input_one in inputs]
    #             image_paths = []
    #             saving_path = "image.jpg"
    #             for image_ in images_list:
    #                 image_.save(saving_path)
    #                 image_paths.append([saving_path])
                
    #             #print(image_paths)
    #             #print(images_list)
    #             with torch.autocast(device_type='cuda', dtype=torch.float16):
    #                 # print reward functions device
    #                 print("reward function device", reward_func.device)

    #                 rewards_per_func[:, i] = torch.FloatTensor(reward_func.get_scores(texts, image_paths, hd_num=2)).to('cuda:1') # Shape (B*G,)
    #             print("reward per func", rewards_per_func)


    #             #print(rewards_per_func.shape)
    #             #print(rewards_per_func)
    #         else:
    #             # Repeat all input columns (but "prompt" and "completion") to match the number of generations
    #             ##print("iam here", input.keys())
    #             keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
    #             reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
    #             output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    #             rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    #     # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
    #     # completions may be distributed across processes
    #     rewards_per_func = gather(rewards_per_func)

    #     # Sum the rewards from all reward functions
    #     rewards = rewards_per_func.sum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    #     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    #     # Slice to keep only the local part of the data
    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )
    #     advantages = advantages[process_slice]

    #     # Log the metrics
    #     reward_per_func = rewards_per_func.mean(0)
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
    #             reward_func_name = reward_func.config._name_or_path.split("/")[-1]
    #         else:
    #             reward_func_name = reward_func.__name__
    #         self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

    #     self._metrics["reward"].append(rewards.mean().item())
    #     self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

    #     if (
    #         self.log_completions
    #         and self.state.global_step % self.args.logging_steps == 0
    #         and "wandb" in self.args.report_to
    #     ):
    #         import pandas as pd

    #         # For logging
    #         table = {
    #             "step": [str(self.state.global_step)] * len(rewards),
    #             "prompt": gather_object(prompts_text),
    #             "completion": gather_object(completions_text),
    #             "reward": rewards.tolist(),
    #         }
    #         df = pd.DataFrame(table)

    #         if wandb.run is not None and self.accelerator.is_main_process:
    #             wandb.log({"completions": wandb.Table(dataframe=df)})

    #     return {
    #         "prompt_ids": prompt_ids,
    #         "prompt_mask": prompt_mask,
    #         "pixel_values": pixel_values,
    #         "completion_ids": completion_ids,
    #         "completion_mask": completion_mask,
    #         "ref_per_token_logps": ref_per_token_logps,
    #         "advantages": advantages,
    #     }

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     if return_outputs:
    #         raise ValueError("The GRPOTrainer does not support returning outputs")
    #     # Compute the per-token log probabilities for the model
    #     ##print(inputs.keys())

    #     prompt_ids, prompt_mask, pixel_values = inputs["prompt_ids"], inputs["prompt_mask"], inputs["pixel_values"]
    #     completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    #     input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    #     logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    #     per_token_logps = self._get_per_token_logps(model, input_ids, pixel_values, logits_to_keep, attention_mask)

    #     # Compute the KL divergence between the model and the reference model
    #     ref_per_token_logps = inputs["ref_per_token_logps"]
    #     per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    #     # x - x.detach() allows for preserving gradients from x
    #     advantages = inputs["advantages"]
    #     per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    #     per_token_loss = -(per_token_loss - self.beta * per_token_kl)
    #     loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    #     # Log the metrics
    #     completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
    #     self._metrics["completion_length"].append(completion_length)

    #     mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    #     self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    #     return loss


    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     if return_outputs:
    #         raise ValueError("GRPOTrainer does not support returning outputs")

    #     #print(inputs.keys())
    #     prompt_input_ids = inputs["prompt_ids"]
    #     pixel_values = inputs["pixel_values"]
    #     prompts = inputs["prompt"]

    #     # Prepare generation inputs
    #     generation_inputs = {
    #         "input_ids": prompt_input_ids,
    #         "pixel_values": pixel_values,
    #         "attention_mask": (prompt_input_ids != self.processing_class.tokenizer.pad_token_id).int(),
    #     }
    #     if "pixel_attention_mask" in inputs:
    #         generation_inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
    #     if "image_sizes" in inputs:
    #         generation_inputs["image_sizes"] = inputs["image_sizes"]

    #     # Generate completions
    #     with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
    #         prompt_completion_ids = unwrapped_model.generate(
    #             **generation_inputs,
    #             generation_config=self.generation_config
    #         )

    #     # Get the per-token log probabilities for the completions 
    #     # (now handles vision-language model inputs)
    #     def _get_per_token_logps(self, model, input_ids, pixel_values, logits_to_keep, attention_mask=None,  **kwargs):
    #         # Forward pass with both text and image inputs
    #         # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    #         outputs = model( #TODO bring back the normal version if passed to a text only model
    #             input_ids=input_ids,
    #             pixel_values=pixel_values,
    #             attention_mask=attention_mask,
    #             logits_to_keep=logits_to_keep + 1,
    #             **kwargs
    #         )
    #         logits = outputs.logits  # (B, L, V)
    #         logits = logits[:, :-1, :]  #exclude the last logit: it corresponds to the next token pred
    #         input_ids = input_ids[:, -logits_to_keep:]  # Remove first token
            

    #         logits = logits[:, -logits_to_keep:]
    #         # # Calculate per-token log probabilities
    #         # log_probs = logits.log_softmax(dim=-1)
    #         # per_token_logps = torch.gather(
    #         #     log_probs, 
    #         #     dim=-1, 
    #         #     index=input_ids.unsqueeze(-1)
    #         # ).squeeze(-1)
            
    #         # return per_token_logps
    #         return selective_log_softmax(logits, input_ids)

    #     # Calculate log probabilities for current policy
    #     per_token_logps = get_per_token_logps(
    #         model,
    #         prompt_completion_ids,
    #         pixel_values=pixel_values.repeat_interleave(self.num_generations, dim=0),
    #         attention_mask=generation_inputs.get("attention_mask")
    #     )

    #     # Calculate reference model log probabilities
    #     with torch.inference_mode():
    #         if self.ref_model is not None:
    #             ref_per_token_logps = get_per_token_logps(
    #                 self.ref_model,
    #                 prompt_completion_ids,
    #                 pixel_values=pixel_values.repeat_interleave(self.num_generations, dim=0),
    #                 attention_mask=generation_inputs.get("attention_mask")
    #             )
    #         else:
    #             # Handle PEFT case by disabling adapters
    #             with model.disable_adapter():
    #                 ref_per_token_logps = get_per_token_logps(
    #                     model,
    #                     prompt_completion_ids,
    #                     pixel_values=pixel_values.repeat_interleave(self.num_generations, dim=0),
    #                     attention_mask=generation_inputs.get("attention_mask")
    #                 )

    #     # Remove prompt parts from log probabilities
    #     prompt_length = prompt_input_ids.size(1)
    #     per_token_logps = per_token_logps[:, prompt_length-1:]
    #     ref_per_token_logps = ref_per_token_logps[:, prompt_length-1:]

    #     # Calculate KL divergence per token
    #     per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
    #                   (ref_per_token_logps - per_token_logps) - 1

    #     # Create completion mask (ignore padding and post-EOS tokens)
    #     completion_ids = prompt_completion_ids[:, prompt_length:]
    #     is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
    #     eos_positions = is_eos.int().argmax(dim=1, keepdim=True)
    #     sequence_lengths = torch.where(
    #         is_eos.any(dim=1),
    #         eos_positions.squeeze(-1) + 1,
    #         torch.tensor(completion_ids.size(1), device=device)
    #     )
    #     completion_mask = torch.arange(
    #         completion_ids.size(1), device=device
    #     )[None, :] < sequence_lengths[:, None]

    #     # Decode completions for reward calculation
    #     completions = self.processing_class.tokenizer.batch_decode(
    #         completion_ids,
    #         skip_special_tokens=True
    #     )

    #     # Prepare inputs for reward functions
    #     batch_size = len(prompts)
    #     expanded_images = pixel_values.repeat_interleave(self.num_generations, dim=0)
    #     expanded_prompts = [p for p in prompts for _ in range(self.num_generations)]

    #     # Calculate rewards using all reward functions
    #     rewards_per_func = torch.zeros(len(expanded_prompts), len(self.reward_funcs), device=device)
        
    #     for i, (reward_func, reward_processor) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
    #         if isinstance(reward_func, PreTrainedModel):
    #             # Handle vision-language reward models
    #             reward_inputs = reward_processor(
    #                 text=[p + c for p, c in zip(expanded_prompts, completions)],
    #                 images=expanded_images,
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=self.max_prompt_length + self.max_completion_length,
    #                 add_special_tokens=False
    #             ).to(device)

    #             with torch.inference_mode():
    #                 rewards = reward_func(**reward_inputs).logits.squeeze(-1)
    #         else:
    #             # Custom reward function handling
    #             rewards = torch.tensor(
    #                 reward_func(
    #                     prompts=expanded_prompts,
    #                     completions=completions,
    #                     images=expanded_images
    #                 ),
    #                 device=device
    #             )
            
    #         rewards_per_func[:, i] = rewards

    #     # Normalize rewards across groups
    #     total_rewards = rewards_per_func.sum(dim=1)
    #     group_rewards = total_rewards.view(batch_size, self.num_generations)
    #     mean_rewards = group_rewards.mean(dim=1, keepdim=True)
    #     std_rewards = group_re wards.std(dim=1, keepdim=True) + 1e-8
    #     normalized_rewards = (group_rewards - mean_rewards) / std_rewards
    #     advantages = normalized_rewards.view(-1, 1)

    #     # Calculate final loss with KL penalty
    #     policy_advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
    #     kl_penalized_loss = (policy_advantages - self.beta * per_token_kl) * completion_mask
    #     normalized_loss = kl_penalized_loss.sum(dim=1) / completion_mask.sum(dim=1)
    #     final_loss = -normalized_loss.mean()

    #     # Log metrics
    #     self._metrics["reward"].append(total_rewards.mean().item())
    #     self._metrics["kl"].append(per_token_kl.mean().item())
    #     self._metrics["completion_length"].append(completion_mask.sum(dim=1).float().mean().item())

    #     return final_loss

    