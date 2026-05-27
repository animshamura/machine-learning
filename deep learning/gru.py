from tensorflow.keras.layers import GRU

model = Sequential([
    Embedding(10000, 128),
    GRU(64),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_split=0.2)
