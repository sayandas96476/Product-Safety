
* A Kivy frontend app
* A FastAPI backend hosted on Hugging Face Spaces
* Image-based ingredient detection
* LangChain + Groq API for reasoning
* Hugging Face Inference API for OCR

---

```markdown
# 🧠 Ingredient Analyzer App

This project is a GenAI-powered ingredient safety app that lets users upload food/beauty product images and get potential health risks based on ingredients detected.

---

## 📱 Frontend (Kivy App)

A Kivy-based Android app allows users to:
- Capture or upload an image of a product
- Select the product category (e.g., food, skincare)
- Submit to the backend for analysis

---

## 🚀 Backend (FastAPI on Hugging Face Space)

The backend performs:
1. **OCR** using Hugging Face's `Qwen2.5-VL-7B-Instruct` model to extract ingredient text from the image.
2. **Reasoning** with LangChain and Groq LLM to:
   - Interpret and refine the OCR result
   - Predict potential diseases or harmful effects
3. **Returns** a health risk summary for the ingredients.

---

## 🛠️ Technologies Used

| Component        | Stack                                     |
|------------------|-------------------------------------------|
| Frontend         | Kivy (Python)                             |
| Backend API      | FastAPI                                   |
| OCR              | Hugging Face Inference API                |
| Reasoning        | LangChain + Groq API                      |
| Hosting (API)    | Hugging Face Spaces                       |
| File I/O         | UploadFile (FastAPI), Base64 encoding     |

---

## 📂 Directory Structure

```

.
├── app.py                # FastAPI backend
├── config.ini            # Keys, prompts, and model configs
├── main.py               # Kivy App
├── ai\_app.py             # Kivy App Logic using FastAPI endpoint
├── requirements.txt
└── README.md

````

---

## ⚙️ config.ini Example

```ini
[groq]
api_key = your_groq_api_key
model_name = Mixtral-8x7B-Instruct-v0.1

[huggingface]
api_key = your_hf_token

[prompts]
fine_prompt = Citing the report in the CONTEXT provided. Can you explain the ingredient impact for the given product type: {product}? CONTEXT: {context}
disease_prompt = Based on the analysis, list possible diseases the ingredient(s) might cause. CONTEXT: {context}
ocr_instruction = First identify if the image contains any ingredient or not. If not, return nothing. Else extract all the ingredients from the image.
````

---

## 🔗 API Endpoint

### `POST /generate_answer`

| Field   | Type | Required | Description                         |
| ------- | ---- | -------- | ----------------------------------- |
| image   | file | ✅        | Image of the product ingredients    |
| product | form | ✅        | Product category (e.g., food, skin) |

#### ✅ Example Response:

```json
{
  "answer": "The ingredient list contains 'Parabens' which may cause hormone disruption and skin irritation over time."
}
```

---

## 📦 Installation (Local Dev)

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

---

## 📱 Kivy App Usage

1. Ensure your Android app (built using Kivy/Buildozer) sends a `POST` request to the Hugging Face FastAPI URL.
2. Use `requests` in `ai_app.py` to call the endpoint:

```python
files = {"image": open(image_path, "rb")}
data = {"product": selected_product}
response = requests.post("https://your-space-name.hf.space/generate_answer", files=files, data=data)
```

---

## 🧪 Test It Online

Deploy your FastAPI backend on [Hugging Face Spaces](https://huggingface.co/spaces) with `sdk: docker` or `sdk: gradio` and `app.py` as entrypoint.

---

## 📜 License

MIT License

---

## 🙋‍♂️ Author

Made with ❤️ by Sayan Das
