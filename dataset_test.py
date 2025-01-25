import torch
# ds = load_dataset("zhiqings/LLaVA-Human-Preference-10K")
from transformers import default_data_collator

from datasets import features, load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch
from trl import RLOOConfig, RLOOTrainer, DPOTrainer, DPOConfig
from peft import LoraConfig
import wandb
import random
from transformers import default_data_collator
from qwen_vl_utils import process_vision_info

def evaluate_model(trainer, eval_dataset, reward_fn, processor):
    """Evaluate the model using a random reward function."""
    results = []
    for example in eval_dataset:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        # Calculate rewards
        chosen_reward = reward_fn(prompt, chosen)
        rejected_reward = reward_fn(prompt, rejected)
        
        # Compute metrics
        reward_gap = chosen_reward - rejected_reward
        results.append({"chosen_reward": chosen_reward, "rejected_reward": rejected_reward, "reward_gap": reward_gap})
    
    # Log metrics to wandb
    avg_chosen_reward = sum(r["chosen_reward"] for r in results) / len(results)
    avg_rejected_reward = sum(r["rejected_reward"] for r in results) / len(results)
    avg_reward_gap = sum(r["reward_gap"] for r in results) / len(results)
    
    wandb.log({
        "evaluation/avg_chosen_reward": avg_chosen_reward,
        "evaluation/avg_rejected_reward": avg_rejected_reward,
        "evaluation/avg_reward_gap": avg_reward_gap,
    })
    return avg_reward_gap



def dummy_reward(prompt, response):
    """Temporary reward function."""
    return random.uniform(0, 1)


def main():

    wandb.init(
        project="rloo_training",
        name="idefics2-8b-rloo",
        #config={"epochs": 1, "batch_size": 2}  # Add other hyperparameters here
    )
    # Load the trainable policy (model) and processor
    model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16)
    ref_model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16)

    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", do_image_splitting=False)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Load the reward model (replace with appropriate reward model)
    reward_model =  AutoModelForSequenceClassification.from_pretrained("cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr", torch_dtype=torch.bfloat16)

    peft_config = LoraConfig(target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],  # Target specific Linear layers
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and preprocess the dataset
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:1%]")

    def format2(example):
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
        
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        chosen = processor.apply_chat_template(chosen, tokenize=False)
        rejected = processor.apply_chat_template(rejected, tokenize=False)

        prompt_input = processor(prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        chosen_input = processor(chosen, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        rejected_input = processor(rejected, padding="max_length", truncation=True, max_length=128, return_tensors="pt")



        #max_size = processor.image_processor.size["longest_edge"] // 2
        max_size = 240
        example["image"].thumbnail((max_size, max_size))
        return {
            "images": [example["image"]],
            "prompt": prompt_input,
            "chosen": chosen_input,
            "rejected": rejected_input
            #"reward": dummy_reward(prompt, chosen) - dummy_reward(prompt, rejected)  # Example reward computation
        }

    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing
        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch


    def format(example):
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
        
        # Apply the processor (it will tokenize text and handle image processing)
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        chosen = processor.apply_chat_template(chosen, tokenize=False)
        rejected = processor.apply_chat_template(rejected, tokenize=False)

        # Tokenize using the processor
        prompt_input = processor(prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        chosen_input = processor(chosen, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        rejected_input = processor(rejected, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        # Make sure the images are preprocessed
        max_size = 240
        example["image"].thumbnail((max_size, max_size))
        print(example["image"])

        return {
            "images": [example["image"]],
            "prompt_input_ids": prompt_input["input_ids"].squeeze().tolist(),
            "prompt_attention_mask": prompt_input["attention_mask"].squeeze().tolist(),
            "chosen_input_ids": chosen_input["input_ids"].squeeze().tolist(),
            "chosen_attention_mask": chosen_input["attention_mask"].squeeze().tolist(),
            "rejected_input_ids": rejected_input["input_ids"].squeeze().tolist(),
            "rejected_attention_mask": rejected_input["attention_mask"].squeeze().tolist(),
        }



    dataset = dataset.map(format2, remove_columns=dataset.column_names, num_proc=32)


    # Make sure that the images are decoded, it prevents from storing bytes.
    # More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)
    print("set")


        # Define RLOO training arguments
    training_args = RLOOConfig(
        output_dir="idefics2-8b-rloo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=2,
        dataset_num_proc=32,
        dataloader_num_workers=2,
        logging_steps=1,
        report_to="wandb"  
    )

    print("config")
    
    # Initialize the RLOO Trainer
    trainer = RLOOTrainer(
        config=training_args,
        policy=model,
        ref_policy=ref_model,  # Optional reference policy
        reward_model=reward_model,
        train_dataset=dataset,
        #peft_config=LoraConfig(target_modules="all-linear"),
        #processing_class=processor,
        processing_class = processor,
        tokenizer = processor.tokenizer,
        #tokenizer = tokenizer,
        data_collator = collate_fn
        #peft_config = peft_config
    )

    # Train the model
    trainer.train()

    # Train the model
    # training_args = DPOConfig(
    #     output_dir="idefics2-8b-dpo",
    #     bf16=True,
    #     gradient_checkpointing=True,
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=32,
    #     num_train_epochs=2,
    #     dataset_num_proc=32,  # tokenization will use 32 processes
    #     dataloader_num_workers=32,  # data loading will use 32 workers
    #     logging_steps=1,
    #     report_to="wandb"
    # )
    # trainer = DPOTrainer(
    #     model,
    #     ref_model=None,  # not needed when using peft
    #     args=training_args,
    #     train_dataset=dataset,
    #     tokenizer=processor,
    #     peft_config=LoraConfig(target_modules="all-linear"),
    # )

    # trainer.train()







    # # Evaluate during or after training
    # eval_dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="validation")  # Load evaluation split
    # eval_dataset = eval_dataset.map(format, remove_columns=eval_dataset.column_names, num_proc=32)

    # average_gap = evaluate_model(trainer, eval_dataset, dummy_reward, processor)
    # print(f"Average Reward Gap: {average_gap}")




    # """Evaluate both models"""
    # test_dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="test")
    # processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
    # reward_model = AutoModelForVision2Seq.from_pretrained("path/to/reward-model", torch_dtype=torch.bfloat16)

    # dpo_score = evaluate_model("idefics2-8b-dpo", processor, test_dataset, reward_model)
    # rloo_score = evaluate_model("idefics2-8b-rloo", processor, test_dataset, reward_model)

    # print(f"DPO Model Score: {dpo_score}")
    # print(f"RLOO Model Score: {rloo_score}")


if __name__ == "__main__":
    main()
