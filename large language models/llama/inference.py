from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "meta-llama/Llama-2-13b-chat-hf"  # You can choose 7b/13b/70b depending on your setup

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Explain the difference between supervised and unsupervised learning with code examples."

output = pipe(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9)[0]['generated_text']
print(output)
