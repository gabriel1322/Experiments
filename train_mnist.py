import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
import tensorflow_federated as tff
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np

random.seed(1234)
tf.random.set_seed(1234)

# Argument Parsing
parser = argparse.ArgumentParser(description='Train a model with fairness, robustness, and differential privacy considerations.')

parser.add_argument('--NUM_CLIENTS', type=int, default=10, help='Number of clients')
parser.add_argument('--BATCH_SIZE', type=int, default=16, help='Batch size')
parser.add_argument('--EPOCHS', type=int, default=10, help='Number of epochs')
parser.add_argument('--fairness_parameter', type=float, default=0, help='Fairness parameter value (lambda used for regularization)')
parser.add_argument('--l2_norm_clip', type=float, default=1e30, help='L2 norm clip for DP (gradient clipping)')
parser.add_argument('--noise_multiplier', type=float, default=0, help='Noise multiplier for DP (gradient noise))')
parser.add_argument('--number_versions', type=float, default=0, help='Number of versions for each sample (randomized smoothing)')
parser.add_argument('--noise_scale', type=float, default=1, help='Noise scale (randomized smoothing)')
parser.add_argument('--noise_test', type=float, default=0.2, help='Robustness parameter value (noise added to input data)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for DP optimizer')
args = parser.parse_args()

# Preparing the input data
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch of EMNIST data and return a (features, label) tuple."""
    return (tf.reshape(element['pixels'], [-1, 784]), 
            tf.reshape(element['label'], [-1, 1]))

  return dataset.batch(args.BATCH_SIZE).map(batch_format_fn)

client_ids = sorted(emnist_train.client_ids)[:args.NUM_CLIENTS]
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x))
  for x in client_ids
]

def transform_noisy(image, label):
  image_noisy = image + tf.random.normal(shape=tf.shape(image), mean=0, stddev=args.noise_test, dtype=tf.float32) 
  return (image_noisy, label)
        
# Add gaussian noise to a dataset of a client
def add_gaussian_noise(dataset):
  return dataset.map(transform_noisy)

# Add gaussian noise to all the datasets of the clients
#noisy_federated_train_data = []
#for dataset in federated_train_data:
    #new_dataset = add_gaussian_noise(dataset)
    #noisy_federated_train_data.append(new_dataset)
#federated_train_data = noisy_federated_train_data

# Let's print the first image of the first client
#first_client_dataset = list(federated_train_data[1].as_numpy_iterator())
#first_image, first_label = first_client_dataset[1]

#plt.imshow(first_image.reshape(28, 28), cmap='gray')  # assuming the image is 28x28
#plt.title(f"Label: {first_label[0]}")
#plt.show()

# Model creation : single hidden layer, followed by a softmax layer.
def create_keras_model():
  initializer = tf.keras.initializers.GlorotNormal(seed=0)
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer=initializer),
      tf.keras.layers.Softmax(),
  ])

# Define fairness term
def fairness_term(predictions):
  rates = tf.reduce_mean(predictions, axis=0)
  dp_loss = tf.math.reduce_variance(rates)
  fairness = args.fairness_parameter * dp_loss
  return fairness

# In order to use this model in TFF, wrap the Keras model as a tff.learning.models.VariableModel
def model_fn():
  keras_model = create_keras_model()
  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec=federated_train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
  """Performs training (using the server model weights) on the client's dataset."""
  # Initialize the client model with the current server weights.
  client_weights = model.trainable_variables
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)

  # Use the client_optimizer to update the local model.
  for batch in dataset:
      # Compute a forward pass on the batch of data
      outputs = model.forward_pass(batch)
      custom_loss = outputs.loss + fairness_term(outputs.predictions)
      # Compute the corresponding gradient
      grads = client_optimizer.compute_gradients(custom_loss, client_weights)

      # Apply the gradient using a client optimizer.
      client_optimizer.apply_gradients(grads)

  return client_weights

@tf.function
def server_update(model, mean_client_weights):
  """Updates the server model weights as the average of the client model weights."""
  model_weights = model.trainable_variables
  # Assign the mean client weights to the server model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        model_weights, mean_client_weights)
  return model_weights

# Initialize function 
@tff.tf_computation
def server_init():
  model = model_fn()
  return model.trainable_variables

# Passing this into a federated computation using tff.federated_value.
@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)

whimsy_model = model_fn()
tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)
model_weights_type = server_init.type_signature.result

@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
  model = model_fn()
  client_optimizer = DPAdamGaussianOptimizer(
      l2_norm_clip=args.l2_norm_clip,
      noise_multiplier=args.noise_multiplier,
      num_microbatches=1,
      learning_rate=args.learning_rate)
  return client_update(model, tf_dataset, server_weights, client_optimizer)

@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
  model = model_fn()
  return server_update(model, mean_client_weights)

federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = tff.federated_map(
      client_update_fn, (federated_dataset, server_weights_at_client))

  # The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)

  return server_weights

if __name__ == "__main__":
    federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
    )

    central_emnist_test = emnist_test.create_tf_dataset_from_all_clients()
    central_emnist_test = preprocess(central_emnist_test)
    
    #noisy_central_emnist_test = add_gaussian_noise(central_emnist_test)
    
    def evaluate(server_state):
        #keras_model = tf.keras.models.load_model('mnist-federated_model.h5')
        keras_model = create_keras_model()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
        )
        keras_model.set_weights(server_state)
        #keras_model.evaluate(noisy_central_emnist_test)
        keras_model.evaluate(central_emnist_test)
        keras_model.save('mnist-federated_model_test2.h5')
        #predictions = keras_model.predict(central_emnist_test)
        #predictions_mean = tf.math.reduce_mean(predictions, axis=0)
        #tf.print(predictions_mean)
        #tf.print(tf.math.reduce_variance(predictions_mean))

    def transform_noisy1(image, label):
      image_noisy = image + tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.1*args.noise_scale, dtype=tf.float32) 
      return (image_noisy, label)
    def add_gaussian_noise1(dataset):
      return dataset.map(transform_noisy1)
    
    def transform_noisy2(image, label):
      image_noisy = image + tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.125*args.noise_scale, dtype=tf.float32) 
      return (image_noisy, label)
    def add_gaussian_noise2(dataset):
      return dataset.map(transform_noisy2)
    
    def transform_noisy3(image, label):
      image_noisy = image + tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.15*args.noise_scale, dtype=tf.float32) 
      return (image_noisy, label)
    def add_gaussian_noise3(dataset):
      return dataset.map(transform_noisy3)

    def transform_noisy4(image, label):
      image_noisy = image + tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.175*args.noise_scale, dtype=tf.float32) 
      return (image_noisy, label)
    def add_gaussian_noise4(dataset):
      return dataset.map(transform_noisy4)
    
    def transform_noisy5(image, label):
      image_noisy = image + tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.2*args.noise_scale, dtype=tf.float32) 
      return (image_noisy, label)
    def add_gaussian_noise5(dataset):
      return dataset.map(transform_noisy5)
    
    def randomized_smoothing_predict(dataset, num_samples, confidence_threshold):
      #Unpack the labels and store them in a list
      labels = []
      for element in dataset:
        image, label = element
        for label in label:
          labels.append(label)

      # Map the dataset to a new mapdataset with gaussian noise
      dataset1 = add_gaussian_noise1(dataset)
      dataset2 = add_gaussian_noise2(dataset)
      dataset3 = add_gaussian_noise3(dataset)
      dataset4 = add_gaussian_noise4(dataset)
      dataset5 = add_gaussian_noise5(dataset)

      # Concatenate the datasets
      new_dataset = tf.data.Dataset.concatenate(dataset, dataset1)
      new_dataset = tf.data.Dataset.concatenate(new_dataset, dataset2)
      new_dataset = tf.data.Dataset.concatenate(new_dataset, dataset3)
      new_dataset = tf.data.Dataset.concatenate(new_dataset, dataset4)
      new_dataset = tf.data.Dataset.concatenate(new_dataset, dataset5)
     
      # Randomized smoothing evaluation on the new dataset
      model = tf.keras.models.load_model('mnist-federated_model.h5')
      sum = 0
      count = 0
      predictions = model.predict(new_dataset)
      predictions_samples = []
      for i in range(40832):
        counter = 0
        for j in range(i, i + 244992, 40832): # 6 times in total
            if counter == 0:
              initial_prediction = np.argmax(predictions[j])
              counter += 1
            else: 
              predictions_samples.append(predictions[j])
              counter += 1
              if np.argmax(predictions[j]) == initial_prediction:
                count += 1
              if counter==num_samples:
                confidence = count / num_samples
                if confidence >= confidence_threshold:
                  final_prediction = initial_prediction
                else:
                  final_prediction = np.argmax(tf.math.reduce_mean(predictions_samples, axis=0))
                if final_prediction == labels[i]:
                  sum += 1
                else:
                  sum += 0
      accuracy = sum/40832 
      print('Accuracy: ', accuracy)

    server_state = federated_algorithm.initialize()

    for round in range(args.EPOCHS):
        server_state = federated_algorithm.next(server_state, federated_train_data)
        if round % 10 ==0 :
          evaluate(server_state)

    #evaluate(server_state)
    #randomized_smoothing_predict(noisy_central_emnist_test, 5, 1)

    # Load the model with server weights
    #model = tf.keras.models.load_model('saved_model.h5')

    #first_image, first_label = next(iter(central_emnist_test))

    # Predict
    #probability_vector = model.predict(first_image)
    #print("Probability vector:", tf.nn.softmax(probability_vector[0]))
    #print("Predicted label:", np.argmax(probability_vector[0]))

    #plt.imshow(first_image.numpy().reshape(28, 28), cmap='gray')  # assuming the image is 28x28
    #plt.title(f"Actual Label: {first_label.numpy()[0]}")
    #plt.show()