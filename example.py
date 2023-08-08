import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

# 1. Load the trained model from .h5 checkpoint
model = load_model('cifar10_cnn_model.h5')

# 2. Load one image from the CIFAR-10 test dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
sample_image = test_images[0]  # Take the first image from the test set as an example

# Display the image (optional)
import matplotlib.pyplot as plt
#plot on a tiny window
plt.figure(figsize=(2,2))
plt.imshow(sample_image)
plt.show()

# 3. Pre-process the image
# Assuming your model was trained on images normalized to [0, 1]
sample_image = sample_image.astype('float32') / 255.0  # Normalize
sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension, shape becomes (1, 32, 32, 3)

# 4. Predict and get the probability vector
probability_vector = model.predict(sample_image)
print("Probability vector:", tf.nn.softmax(probability_vector[0]))
print("Predicted label:", np.argmax(probability_vector[0]))
print("correct label : ",test_labels[0])

dict_labels = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

print("Predicted label:", dict_labels[np.argmax(probability_vector[0])])
print("correct label : ",dict_labels[test_labels[0][0]])

