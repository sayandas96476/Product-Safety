from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse




app = FastAPI()

import requests
import os
import os
hf_token = os.getenv("HF_API_KEY")
groq_token = os.getenv("GROQ_API_KEY")

import re
import base64
import mimetypes
from huggingface_hub import InferenceClient

def image_to_base64(image_path):
    """Convert local image to base64 string"""
    with open(image_path, "rb") as image_file:
        # Read the image file
        image_data = image_file.read()
        # Encode to base64
        base64_string = base64.b64encode(image_data).decode('utf-8')
        
        # Get the MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            # Default to JPEG if we can't determine the type
            mime_type = "image/jpeg"
        
        # Return data URL format
        return f"data:{mime_type};base64,{base64_string}"

def ocr(local_image_path):
    # Convert image to base64
    base64_image = image_to_base64(local_image_path)
    print("line47")

    client = InferenceClient(
        provider="auto",
        api_key=hf_token,
    )

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Step 1: Analyze the image and determine whether it contains an ingredient list or any ingredient-related information.

- If no ingredients are found or the image does not contain relevant data, return an empty result or "No ingredients detected."
- If the image includes ingredients, extract the full list of ingredients accurately.
  - Ensure the extraction is clean, readable, and free from OCR noise.
  - Preserve the order if it appears as a formal list on the packaging.

Your output should only include the ingredients if found.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            }
        ],
    )

    print("Success with local image!")
    #print(completion.choices[0].message)
    return completion.choices[0].message


import requests

class Harmful_check:
    def __init__(self):
        self.api_key = groq_token
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "Deepseek-R1-Distill-Llama-70b"

    def call_groq(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 1024
        }

        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def potential_disease(self, CONTEXT):
        template = """
        Given the CONTEXT which contains extracted ingredients from a grocery product label:

        1. Clean and verify the list of ingredients by correcting OCR errors, standardizing chemical names, and removing duplicates or noise.
        2. For each verified ingredient, provide a detailed clinical insight including:
        - Known health effects (short-term and long-term)
        - Associated diseases or medical conditions (if any)
        - Regulatory status (e.g., FDA-approved, banned in some countries)
        3. If the CONTEXT seems incomplete or unclear, infer likely ingredients based on the product type if possible.
        4. Present your analysis in a structured format.

        CONTEXT: {CONTEXT}
        """

        prompt = template.format(CONTEXT=CONTEXT)
        text = self.call_groq(prompt)
        clean_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        return clean_text

    def Fine(self, CONTEXT, PRODUCT):
        prompt = f"""
You are a Product Safety Officer with expertise in Chemistry and Health Care. Your task is to evaluate the safety of a given product based on its chemical composition.
Instructions:
1. Carefully analyze the chemical composition provided.
2. Think through each component’s properties and potential health effects.
3. Consider interactions between components if relevant.
4. Reason step-by-step and explicitly explain your thought process.
5. Finally, provide a detailed, evidence-based conclusion on whether the product is harmful or safe for human use.
6. Clearly explain your reasoning, referencing relevant chemical or health principles.
CONTEXT: {CONTEXT}
QUESTION: Is the following product safe for human use? Please provide your conclusion along with the reasons.
PRODUCT: {PRODUCT}
Answer with your step-by-step reasoning first (chain of thought), then conclude with a clear yes or no and justification.
"""
        text = self.call_groq(prompt)
        clean_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        return clean_text



@app.post("/generate_answer")
async def generate_answer(
    image: UploadFile = File(...),
    product: str = Form(...)
):
    # Save the uploaded file temporarily
    #temp_file_path = f"temp_{image.filename}"
    temp_file_path = f"/tmp/temp_{image.filename}"

    with open(temp_file_path, "wb") as f:
        f.write(await image.read())

    # Run OCR and analysis
    check_obj = Harmful_check()
    text_sunscreen = ocr(temp_file_path)

    answer = check_obj.Fine(text_sunscreen, product)
    final_answer = check_obj.potential_disease(answer)

    # Clean up temp file
    os.remove(temp_file_path)

    return JSONResponse(content={"answer": final_answer})
