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
import matplotlib.pyplot as plt

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
            text=prompts_text, images=[image], return_tensors="pt"
        ).to('cuda')
        
        prompt_ids, prompt_mask, pixel_values = prompt_inputs["input_ids"], prompt_inputs["attention_mask"], prompt_inputs["pixel_values"]

        generated_texts_list = []
        for _ in range(5):
            with torch.no_grad():
                prompt_completion_ids = model.generate(
                    input_ids=prompt_ids, 
                    pixel_values=pixel_values, 
                    attention_mask=prompt_mask,
                    max_new_tokens=512,
                    do_sample=True,
                )
                prompt_length = prompt_ids.size(1)
                completion_ids = prompt_completion_ids[:, prompt_length:]
                generated_text = processor.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
                generated_texts_list.append(chat_version(data['question'], generated_text))

        generated_scores_list = []
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for generated_texts in generated_texts_list:
                generated_score = reward_model.get_score(generated_texts, [image_path], hd_num=2)
                generated_scores_list.append(generated_score)

        avg_generated_score = np.mean(generated_scores_list)
        generated_scores.append(avg_generated_score)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            chosen_score = reward_model.get_score(chosen_texts, [image_path], hd_num=2)
            rejected_score = reward_model.get_score(rejected_texts, [image_path], hd_num=2)

        # # Save scores
        chosen_scores.append(chosen_score)
        rejected_scores.append(rejected_score)

        # Logging detailed information for generated outputs
        generated_log.append({
            "idx": data['idx'],
            "question": data['question'],
            "rejected": rejected_texts,
            "generated_completion": generated_texts_list,
            "chosen": chosen_texts,      
            "rejected_score": rejected_score,
            "generated_score": generated_scores_list,
            "chosen_score": chosen_score,
            "image_path": image_path
        })
        # Delete the image after use
        if nu > 10:
            if os.path.exists(image_path):
                os.remove(image_path)

        # Save intermediate progress
        with open("generated_scores.json", "w") as f:
            json.dump(generated_log, f, indent=4)

        nu += 1
        if nu > 500:
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


    histogram_data = {
    "chosen_scores": chosen_scores,
    "rejected_scores": rejected_scores,
    "generated_scores": generated_scores
    }

    with open(os.path.join(image_save_dir, "histogram_data.json"), "w") as f:
        json.dump(histogram_data, f, indent=4)


    # Calculate the x range to be the same for both plots
    all_scores = chosen_scores + rejected_scores + generated_scores
    x_min = min(all_scores)
    x_max = max(all_scores)
    bins = 50

    # Plotting histograms on two subplots
    plt.figure(figsize=(16, 13))  # 16:9 aspect ratio

    # Plot chosen scores histogram
    plt.subplot(3, 1, 1)
    plt.hist(chosen_scores, bins=bins, range=(x_min, x_max), alpha=0.5, label='Chosen Scores', color='green')
    plt.title('Chosen, Rejected and Generated Scores Histograms')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_tick_params(labelbottom=False)
    plt.legend(loc='upper left')

    # Plot rejected scores histogram
    plt.subplot(3, 1, 2)
    plt.hist(rejected_scores, bins=bins, range=(x_min, x_max), alpha=0.5, label='Rejected Scores', color='red')
    #plt.title('Rejected Scores Histogram')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_tick_params(labelbottom=False)
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.hist(generated_scores, bins=bins, range=(x_min, x_max), alpha=0.5, label='Generated Scores', color='blue')
    #plt.title('Generated Scores Histogram')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the image
    image_save_dir = '/dss/dsshome1/03/ra59zom2/vlm_rlf/vlm-rl'  # Update this to your desired path
    output_path = os.path.join(image_save_dir, "score_histograms2.png")
    plt.savefig(output_path, dpi=300)  # High quality
    plt.close()

    print(f"Histogram saved at: {output_path}")

if __name__ == "__main__":
    main()
