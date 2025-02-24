import torch
import json
import os
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor,
    AutoModel,
    AutoTokenizer
)
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import numpy as np

def main():
    # Initialize WandB
    wandb.init(project="reward_model_analysis", name="score_histograms")

    # Model initialization
    model_name = "HuggingFaceM4/idefics2-8b"
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)

    # Reward model setup
    reward_model = AutoModel.from_pretrained(
        "internlm/internlm-xcomposer2d5-7b-reward", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model.config._name_or_path, trust_remote_code=True)
    reward_model.tokenizer = reward_tokenizer

    # PEFT configuration
    # peft_config = LoraConfig(
    #     target_modules="all-linear"
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # Dataset preparation
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:1%]")

    # Image saving directory
    image_save_dir = "saved_images"
    os.makedirs(image_save_dir, exist_ok=True)

    # Initialize logs and scores
    chosen_scores = []
    rejected_scores = []
    generated_scores = []
    generated_log = []

    def chat_version(prompt, completion):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
    nu = 0
    
    # Processing dataset
    for data in tqdm(dataset, desc="Processing Dataset"):
        # Prepare texts for chosen and rejected
        chosen_texts = chat_version(data['question'], data['chosen'])
        rejected_texts = chat_version(data['question'], data['rejected'])
        
        # Save image and get path
        image = data["image"]
        image_path = os.path.join(image_save_dir, f"{data['idx']}.jpg")
        image.save(image_path)

        # Generate output using idefics2 model
        prompt = data['question']
        message_list = [[{"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]}]]
        prompts_text = processor.apply_chat_template(message_list, add_generation_prompt=True)
        prompt_inputs = processor(
            text=prompts_text, images=[image], return_tensors="pt", padding=True, 
            padding_side="left", add_special_tokens=False
        ).to('cuda')
        
        prompt_ids, prompt_mask, pixel_values = prompt_inputs["input_ids"], prompt_inputs["attention_mask"], prompt_inputs["pixel_values"]

        with torch.no_grad():
            prompt_completion_ids = model.generate(
                input_ids=prompt_ids, 
                pixel_values=pixel_values, 
                attention_mask=prompt_mask
            )

        # Extract completion ids and decode
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        generated_text = processor.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]

        # Prepare for reward scoring
        generated_texts = chat_version(data['question'], generated_text)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            chosen_score = reward_model.get_score(chosen_texts, [image_path], hd_num=2)
            rejected_score = reward_model.get_score(rejected_texts, [image_path], hd_num=2)
            generated_score = reward_model.get_score(generated_texts, [image_path], hd_num=2)

        # # Save scores
        chosen_scores.append(chosen_score)
        rejected_scores.append(rejected_score)
        generated_scores.append(generated_score)

        # Logging detailed information for generated outputs
        generated_log.append({
            "question": data['question'],
            "rejected": rejected_texts,
            "generated_completion": generated_text,
            "chosen": chosen_texts,      
            "rejected_score": rejected_score,
            "generated_score": generated_score,
            "chosen_score": chosen_score,
            "image_path": image_path
        })

        # Save intermediate progress
        with open("generated_scores.json", "w") as f:
            json.dump(generated_log, f, indent=4)

        nu += 1
        if nu > 5:
            break

    # Calculate averages
    avg_chosen_score = np.mean(chosen_scores)
    avg_rejected_score = np.mean(rejected_scores)
    avg_generated_score = np.mean(generated_scores)

    # Log averages
    with open("average_scores.json", "w") as f:
        json.dump({
            "average_chosen_score": avg_chosen_score,
            "average_rejected_score": avg_rejected_score,
            "average_generated_score": avg_generated_score
        }, f, indent=4)

    # WandB Histograms
    wandb.log({"Chosen Scores Histogram": wandb.Histogram(chosen_scores)})
    wandb.log({"Rejected Scores Histogram": wandb.Histogram(rejected_scores)})
    wandb.log({"Generated Scores Histogram": wandb.Histogram(generated_scores)})

    # Create a table with all scores
    data = list(zip(chosen_scores, rejected_scores, generated_scores))
    table = wandb.Table(data=np.asarray(data), columns=["chosen", "rejected", "generated"])

    # Log overlapping histograms
    wandb.log({
        'overlapping_histogram': wandb.plot.histogram(table, "chosen", title="Scores Histogram"),
        'overlapping_histogram_2': wandb.plot.histogram(table, "rejected", title=None),
        'overlapping_histogram_3': wandb.plot.histogram(table, "generated", title=None)
    })

    print("Processing complete. Results saved to JSON files and WandB.")

if __name__ == "__main__":
    main()
