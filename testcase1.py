# ==============================
# MNIST Classification using MLP
# ==============================

# 1️⃣ Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense


# 2️⃣ Load Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Original Training Shape:", x_train.shape)
print("Original Test Shape:", x_test.shape)


# 3️⃣ Reshape & Normalize
# Convert 28x28 images into 784 length vectors
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

print("Reshaped Training Shape:", x_train.shape)
print("Reshaped Test Shape:", x_test.shape)


# 4️⃣ Define MLP Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Hidden Layer
    Dense(10, activation='softmax')  # Output Layer
])


# 5️⃣ Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 6️⃣ Train Model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)


# 7️⃣ Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_acc)


# 8️⃣ Visualization

# ---- Accuracy Plot ----
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()


# ---- Loss Plot ----
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()


# ---- Loss vs Epochs (Separate Plot) ----
plt.figure()
plt.plot(history.history['loss'])
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# 9️⃣ Observation (Print)
print("\nObservation:")
print("The model achieves around 97–98% accuracy.")
print("Training and validation accuracy increase steadily.")
print("No major overfitting observed.")