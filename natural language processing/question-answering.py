from transformers import pipeline

qa = pipeline("question-answering")
context = "The capital of France is Paris. It is known for the Eiffel Tower."
question = "What is the capital of France?"
result = qa(question=question, context=context)
print(result['answer'])
