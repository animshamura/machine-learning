def unet_model(input_size=(128, 128, 1)):
    inputs = tf.keras.Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D()(conv2)

    up1 = tf.keras.layers.UpSampling2D()(conv2)
    concat1 = tf.keras.layers.concatenate([conv1, up1])
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv3)
    return tf.keras.Model(inputs, outputs)

# Compile and train
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs=10, validation_split=0.1)
