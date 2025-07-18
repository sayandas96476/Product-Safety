from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import re
import base64
import mimetypes
import configparser
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ----------- Load Configuration -----------
config = configparser.ConfigParser()
config.read("config.ini")

# API Keys
groq_api_key = config["groq"]["api_key"]
groq_model = config["groq"]["model_name"]
hf_token = config["huggingface"]["api_key"]

# Prompts
fine_prompt_text = config["prompts"]["fine_prompt"]
disease_prompt_text = config["prompts"]["disease_prompt"]
ocr_instruction_text = config["prompts"]["ocr_instruction"]

# ----------- FastAPI Setup -----------
app = FastAPI()

# ----------- Image to Base64 -----------
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_string = base64.b64encode(image_data).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/jpeg"
        return f"data:{mime_type};base64,{base64_string}"

# ----------- OCR Using Hugging Face Model -----------
def ocr(local_image_path):
    base64_image = image_to_base64(local_image_path)
    client = InferenceClient(api_key=hf_token)

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ocr_instruction_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image}
                    }
                ]
            }
        ]
    )

    return completion.choices[0].message["content"]

# ----------- LangChain LLM Setup -----------
llm = ChatGroq(api_key=groq_api_key, model_name=groq_model)

fine_prompt = PromptTemplate(input_variables=["context", "product"], template=fine_prompt_text)
disease_prompt = PromptTemplate(input_variables=["context"], template=disease_prompt_text)

fine_chain = LLMChain(llm=llm, prompt=fine_prompt)
disease_chain = LLMChain(llm=llm, prompt=disease_prompt)

# ----------- FastAPI Endpoint -----------
@app.post("/generate_answer")
async def generate_answer(image: UploadFile = File(...), product: str = Form(...)):
    temp_file_path = f"/tmp/temp_{image.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await image.read())

    try:
        # Step 1: OCR
        ingredient_text = ocr(temp_file_path)

        # Step 2: LangChain evaluation
        fine_result = fine_chain.run(context=ingredient_text, product=product)
        disease_result = disease_chain.run(context=fine_result)

        # Clean up tags
        final_result = re.sub(r"<think>.*?</think>\s*", "", disease_result, flags=re.DOTALL)

        return JSONResponse(content={"answer": final_result})
    finally:
        os.remove(temp_file_path)
