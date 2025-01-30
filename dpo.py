"""
Tutorial from https://huggingface.co/blog/dpo_vlm
"""


import torch

from datasets import features, load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig
from trl import RLOOConfig, RLOOTrainer, DPOTrainer, DPOConfig

import wandb
import random




def main():

    model_name = "HuggingFaceM4/idefics2-8b"
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16)
   
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)

    #peft_config = LoraConfig(target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    peft_config = LoraConfig(target_modules="all-linear")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ## Handle dataset here!!! They all have different columns
    # Has "image", "question", "rejected", "chosen"
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:3%]")

    
    def format(example):
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
        
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        chosen = processor.apply_chat_template(chosen, tokenize=False)
        rejected = processor.apply_chat_template(rejected, tokenize=False)

        
        max_size = processor.image_processor.size["longest_edge"] // 2
        example["image"].thumbnail((max_size, max_size))
        return {
            "images": [example["image"]],
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    dataset = dataset.map(format, remove_columns=dataset.column_names)

    # Make sure that the images are decoded, it prevents from storing bytes.
    # More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)


    wandb.init(
        project="dpo_training",
        name="idefics2-8b-dpo",
        config={"gradient_acc_steps": 16, "batch_size": 2, "dataset": "2%"}  
        )


    training_args = DPOConfig(
        output_dir="idefics2-8b-dpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        dataset_num_proc=32,  # tokenization will use 32 processes
        dataloader_num_workers=32,  # data loading will use 32 workers
        logging_steps=1,
        report_to="wandb"
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,  # not needed when using peft
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
        peft_config=LoraConfig(target_modules="all-linear"),
    )

    # Train the model
    trainer.train()




if __name__ == "__main__":
    main()
