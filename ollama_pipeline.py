import base64
import requests
from openai import OpenAI
from constants import MODEL, SYSTEM_PROMPT
import os
from dotenv import load_dotenv
load_dotenv()


class OllamaPipeline:

    def __init__(self, provider="ollama", model=MODEL):
        self.model = model
        if provider == 'openai':
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
            )
        elif provider == 'ollama':
            self.client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama',  # required, but unused
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def encode_image_from_url(self, image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # This will raise an exception for HTTP errors
            return base64.b64encode(response.content).decode('utf-8')
        except requests.RequestException as e:
            print(f"Error fetching image from URL: {e}")
            return None

    def analyze_image(self, base64_image, prompt, system_prompt=SYSTEM_PROMPT):
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        },
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
