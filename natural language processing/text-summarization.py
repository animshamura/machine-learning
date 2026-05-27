from transformers import pipeline

summarizer = pipeline("summarization")
text = """
OpenAI has released ChatGPT, a large language model designed for human-like conversation and problem-solving...
"""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])
