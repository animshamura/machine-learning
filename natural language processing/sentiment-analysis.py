from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I absolutely love using Hugging Face Transformers!"))
