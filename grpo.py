import torch
from datasets import features, load_dataset
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    default_data_collator
)
from peft import get_peft_model, LoraConfig
from grpo_trainer import GRPOTrainer
from grpo_config import GRPOConfig
import wandb

def main():
    # Model initialization
    model_name = "HuggingFaceM4/idefics2-8b"
    #model_name = "HuggingFaceTB/SmolVLM-Instruct"
    model_name = "HuggingFaceTB/SmolVLM-256M-Base"
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
    processor.tokenizer.padding_side = "left"
    
    # Reward model setup
    reward_model = AutoModel.from_pretrained(
        "internlm/internlm-xcomposer2d5-7b-reward", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model.config._name_or_path, trust_remote_code=True)
    reward_model.tokenizer = reward_tokenizer
    reward_model.eval()

    # PEFT configuration
    peft_config = LoraConfig(
        target_modules="all-linear"
    )

    # Dataset preparation
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:1%]")

    # Training configuration
    training_args = GRPOConfig(
        output_dir="idefics2-8b-grpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=1e-5,
        logging_steps=1,
        report_to="wandb",
        log_completions=True
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