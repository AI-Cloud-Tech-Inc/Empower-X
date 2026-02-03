import numpy as np
import tensorflow as tf

# Sample data
data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, size=(100,))

# Simple model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
history = model.fit(data, labels, epochs=5, batch_size=8, verbose=1)

# Debugging: Set a breakpoint here to inspect model, data, or metrics
print("Training complete. Accuracy:", history.history['accuracy'][-1])

# You can use the AI/ML Debugger extension to inspect tensors, model architecture, and training metrics at this point.
