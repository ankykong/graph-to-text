import torch
import csv
from constants import PROMPTS, CSV_FILENAME, FIELDNAMES
from transformers import TextStreamer
from unsloth import FastVisionModel


def process_image2(image, image_id):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "ankykong/llama3.2-vision-graphs", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model) # Enable for inference!

    instruction = "You are an expert in reading graphs and understanding the data from it. Describe accurately what you see in this image."

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    # Remove the streamer and get the output directly
    output_ids = model.generate(
        **inputs,
        max_new_tokens = 128,
        use_cache = True,
        temperature = 1.5,
        min_p = 0.1
    )
    
    # Get the input length to skip the prompt
    input_length = inputs.input_ids.shape[1]
    # Decode only the generated part by slicing the output_ids
    result = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)


    # Format results
    line = {
        'Image ID': image_id,
        'Caption': result
    }

    # Append results to CSV
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writerow(line)

    print(f"Results appended to {CSV_FILENAME}")