from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["I love this!", "This is awful", "Such a good experience", "I hate this movie"]
labels = [1, 0, 1, 0]  # binary sentiment

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=10)

model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=10),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=5)
