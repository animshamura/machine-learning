from gensim import corpora, models
from nltk.tokenize import word_tokenize

texts = [["nlp", "is", "fun"], ["deep", "learning", "is", "useful"], ["nlp", "deep", "learning"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)
for idx, topic in lda.print_topics():
    print(f"Topic {idx}: {topic}")
