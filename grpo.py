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
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)

    # Reward model setup
    reward_model = AutoModel.from_pretrained(
        "internlm/internlm-xcomposer2d5-7b-reward", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model.config._name_or_path, trust_remote_code=True)
    reward_model.tokenizer = tokenizer

    # PEFT configuration
    peft_config = LoraConfig(
        target_modules="all-linear"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset preparation
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:1%]")

    # def format_example(example):
    #     # Prepare multimodal prompt
    #     prompt = [{
    #         "role": "user", 
    #         "content": [
    #             {"type": "image"},
    #             {"type": "text", "text": example["question"]}
    #         ]
    #     }]
        
    #     # Process images
    #     example["image"].thumbnail((240, 240))
        
    #     return {
    #         "images": [example["image"]],
    #         "prompt": processor.apply_chat_template(prompt, tokenize=False),
    #         "chosen": example["chosen"],
    #         "rejected": example["rejected"]
    #     }

    # # Process dataset
    # dataset = dataset.map(format_example, remove_columns=dataset.column_names, num_proc=32)
    # dataset = dataset.cast(features.Features({
    #     "images": features.Sequence(features.Image(decode=True)),
    #     "prompt": features.Value("string"),
    #     "chosen": features.Value("string"),
    #     "rejected": features.Value("string")
    # }))
    #TODO bring key setting of the dataset here

    # Training configuration
    training_args = GRPOConfig(
        output_dir="idefics2-8b-grpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=2,
        learning_rate=1e-5,
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