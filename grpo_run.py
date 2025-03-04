import torch
import os
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig
from grpo_trainer import GRPOTrainer
from grpo_config import GRPOConfig

def main():
    # Model initialization
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    #model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False) # This is for idefics
    #processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
   
    # Reward model setup
    reward_model_name = "internlm/internlm-xcomposer2d5-7b-reward"
    reward_model = AutoModel.from_pretrained(
        reward_model_name, 
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
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:5%]")

    # Training configuration
    training_args = GRPOConfig(
        output_dir="idefics2-8b-grpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=1e-5,
        logging_steps=1,
        report_to="wandb",
        log_completions=True
    )
    print(training_args.output_dir)

    # Initialize WandB with all training parameters
    wandb.init(
        project="grpo_training_all_final",
        name="idefics2-8b-grpo",
        config={
            "model": model_name,
            "reward_model": reward_model_name,
            "dataset": "RLAIF-V-Dataset[:5%]",
            "peft": "LoRA",
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "logging_steps": training_args.logging_steps
        }
    )

    # Watch model parameters and gradients
    wandb.watch(model, log_freq=1)

    # Define a callback for logging metrics and saving checkpoints
    class WandbLoggingCallback(TrainerCallback): 
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Log training metrics to WandB."""
            if logs:
                wandb.log(logs)

        def on_step_end(self, args, state, control, **kwargs):
            """Save model checkpoints at 25%, 50%, 75%, and 100% of training."""
            total_steps = state.max_steps
            checkpoint_intervals = [int(total_steps * frac) for frac in [0.25, 0.4, 0.10, 0.5, 0.75, 1.0]]
            checkpoint_intervals.append(1)
        
            print("checkpoint intervals", checkpoint_intervals)
            print(state.global_step)
            
            if state.global_step in checkpoint_intervals:
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{model_name}-{state.global_step}")
                trainer.save_model(checkpoint_dir)
                print(f"Checkpoint saved at step {state.global_step}")

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

    # Add the callback for logging & checkpointing
    trainer.add_callback(WandbLoggingCallback(output_dir=training_args.output_dir))

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
