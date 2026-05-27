import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')
text = "Natural Language Processing is fun and very useful in many applications."

# Tokenization + Stopword Removal
tokens = word_tokenize(text.lower())
filtered = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
print(filtered)

# Named Entity Recognition using SpaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
