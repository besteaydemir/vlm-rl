from transformers import AutoProcessor

model_name = "HuggingFaceM4/idefics2-8b"
processor = AutoProcessor.from_pretrained(model_name)

# Access the tokenizer from the processor
if hasattr(processor, "tokenizer"):
    tokenizer = processor.tokenizer

    # Use the tokenizer's pad method
    batch = [
        {"input_ids": [101, 2009, 2003, 1037, 2060, 102], "attention_mask": [1, 1, 1, 1, 1, 1]},
        {"input_ids": [101, 2023, 2003, 102], "attention_mask": [1, 1, 1, 1]}
    ]
    padded_batch = tokenizer.pad(batch, padding=True, return_tensors="pt")
    print(padded_batch)
