from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

base = ResNet50(include_top=False, input_shape=(224,224,3), weights='imagenet')
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
out = Dense(10, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
