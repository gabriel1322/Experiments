import tensorflow as tf
from datasets import loader
from models import cnn_models

if __name__ == "__main__":
    
    # MNIST
    (train_images, train_labels), (test_images, test_labels) = loader.load_mnist()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    mnist_model = cnn_models.simple_mnist_cnn()
    mnist_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
    mnist_model.fit(train_images, train_labels, epochs=5)
    mnist_model.save('mnist_cnn_model.h5')
    
    # CIFAR10
    (train_images, train_labels), (test_images, test_labels) = loader.load_cifar10()
    train_images = train_images.astype('float32') / 255

    cifar_model = cnn_models.simple_cifar10_cnn()
    cifar_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
    cifar_model.fit(train_images, train_labels, epochs=5)
    cifar_model.save('cifar10_cnn_model.h5')




