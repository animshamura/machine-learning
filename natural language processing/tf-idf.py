from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["I love NLP", "NLP is interesting and fun!", "Deep learning powers NLP"]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)
print(tfidf.get_feature_names_out())
print(X.toarray())
