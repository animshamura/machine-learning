from fastapi import FastAPI, Request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    response = generator(prompt, max_new_tokens=200, temperature=0.7)[0]['generated_text']
    return {"response": response}
