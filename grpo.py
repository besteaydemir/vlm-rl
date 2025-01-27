import torch
from datasets import features, load_dataset
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator
)
from peft import get_peft_model, LoraConfig
from trl import GRPOConfig, GRPOTrainer
import wandb

def main():
    # Model initialization
    model_name = "HuggingFaceM4/idefics2-8b"
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Reward model setup
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr", 
        torch_dtype=torch.bfloat16
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model.config._name_or_path)

    # PEFT configuration
    peft_config = LoraConfig(
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset preparation
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:1%]")

    def format_example(example):
        # Prepare multimodal prompt
        prompt = [{
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["question"]}
            ]
        }]
        
        # Process images
        example["image"].thumbnail((240, 240))
        
        return {
            "images": [example["image"]],
            "prompt": processor.apply_chat_template(prompt, tokenize=False),
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    # Process dataset
    dataset = dataset.map(format_example, remove_columns=dataset.column_names, num_proc=32)
    dataset = dataset.cast(features.Features({
        "images": features.Sequence(features.Image(decode=True)),
        "prompt": features.Value("string"),
        "chosen": features.Value("string"),
        "rejected": features.Value("string")
    }))

    # Training configuration
    training_args = GRPOConfig(
        output_dir="idefics2-8b-grpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=2,
        learning_rate=1e-5,
        max_prompt_length=1024,
        max_completion_length=256,
        num_generations=8,
        beta=0.1,
        remove_unused_columns=False,
        report_to="wandb"
    )

    # Initialize WandB
    wandb.init(
        project="grpo_training",
        name="idefics2-8b-grpo",
        config={
            "model": model_name,
            "dataset": "RLAIF-V-Dataset",
            "peft": "LoRA"
        }
    )

    # Initialize GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_model],
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        reward_processing_classes=[reward_tokenizer],
        peft_config=peft_config
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()