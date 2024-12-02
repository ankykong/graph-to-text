import csv
import time
import subprocess
from ollama_pipeline import OllamaPipeline
from constants import PROMPTS, MODEL, CSV_FILENAME, FIELDNAMES


def process_image(base64_image, image_id):

    provider = "ollama"
    model = "gpt-4o" if provider == "openai" else MODEL
    ollama_pipeline = OllamaPipeline(provider=provider, model=model)
    results = []

    for i, prompt in enumerate(PROMPTS):
        results.append(ollama_pipeline.analyze_image(base64_image, prompt))
        print(f"Analysis completed for prompt {i+1} of {len(PROMPTS)}")

    line = {
        'Image ID': image_id,
        PROMPTS[0]: results[0]}

    if provider == "ollama":
        time.sleep(1)  # Sleep for 1 second after each analysis
        # Run the shell command to stop llava:13b
        subprocess.run(["ollama", "stop", MODEL], check=True)

    # Append results to CSV after each image is processed
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writerow(line)

    print(f"Results appended to {CSV_FILENAME}")
