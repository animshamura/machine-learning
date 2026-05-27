from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM

model_name = "TheBloke/Llama-2-13B-chat-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    use_safetensors=True,
    inject_fused_attention=True,
    trust_remote_code=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "Write a Python program to scrape weather data using BeautifulSoup."

result = pipe(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9)[0]['generated_text']
print(result)
