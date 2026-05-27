from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = (
    "[INST] <<SYS>>\n"
    "You are an expert Python instructor teaching a beginner student.\n"
    "<</SYS>>\n\n"
    "Explain recursion in Python with an analogy and code example. [/INST]"
)

output = pipe(prompt, max_new_tokens=200)[0]['generated_text']
print(output)
