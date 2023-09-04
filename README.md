<!DOCTYPE html>
<html>

<body>

<h1>FRAP: Capturing the Fairness/Robustness/Accuracy/Privacy Trade-Off in Machine Learning Models</h1>

<h2>Description</h2>

<p>
  This repository contains the code and resources for a research project focused on exploring the trade-off between <strong>fairness</strong>,
  <strong>robustness</strong>, <strong>accuracy</strong> and <strong>privacy</strong> in machine learning models through Federated Learning. The experiments were conducted using Python and TensorFlow, leveraging specialized TensorFlow libraries like TensorFlow Privacy and TensorFlow Federated.
</p>

<h2>Table of Contents</h2>

<ul>
  <li><a href="#description">Description</a></li>
  <li><a href="#prerequisites">Prerequisites</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#experiments">Experiments</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
</ul>

<h2>Prerequisites</h2>

<p>     
Before you can run this project, make sure you have met the following requirement:

- Python 3.9+ (Tested on Python 3.10.12)

**Important Note:** TensorFlow Federated supports the following operating systems: Linux and macOS. Please ensure you're running this project on a compatible operating system.

### Installing Required Packages

<ins>**Method 1**</ins>

To install the packages necessary for this project, you can use the `requirements.txt` file. Here's how:

1. Clone this repository to your local machine.
2. Navigate to the project directory using the terminal.
3. If you're using a virtual environment, activate it.
4. Run the following command to install the required packages:
   
   ```bash
   pip install -r requirements.txt

<ins>**Method 2**</ins>

If you encounter any problems with the first method, you can simply install the following dependencies and see if any packages are missing when running the code.
  ```bash
pip install tensorflow
```
```bash
pip install tensorflow_privacy
```
```bash
pip install tensorflow_federated
```
   
</p>

<h2>Usage</h2>

```bash
python train_mnist.py --NUM_CLIENTS [NUM_CLIENTS] --BATCH_SIZE [BATCH_SIZE] --EPOCHS [EPOCHS] --fairness_parameter [fairness_parameter] --l2_norm_clip [l2_norm_clip] --noise_multiplier [noise_multiplier] --number_versions [number_versions] --noise_scale [noise_scale] --noise_test [noise_test] --learning_rate [learning_rate]
```
### Command-line Arguments
<ul>
    <li><b>NUM_CLIENTS:</b> The number of clients participating in the federated learning setup.</li>
    <li><b>BATCH_SIZE:</b> The batch size used for training and evaluation.</li>
    <li><b>EPOCHS:</b> The number of training epochs. The local model weights are aggregated to the server once per epoch.</li>
    <li><b>fairness_parameter:</b> A parameter used for controlling the fairness regularization of the model. The bigger it is, the fairer the model.</li>
    <li><b>l2_norm_clip:</b> The L2 norm gradient clipping parameter for differential privacy.</li>
    <li><b>noise_multiplier:</b> The gradient noise multiplier parameter for differential privacy.</li>
    <li><b>number_versions:</b> The number of versions used for each sample, as part of the randomized smoothing.</li>
    <li><b>noise_scale:</b> The scale of noise applied to versions in randomized smoothing.</li>
    <li><b>noise_test:</b> A parameter used for controlling the noise added to the test dataset. Only used to test the robustness of the model.</li>
    <li><b>learning_rate:</b> The learning rate for the training process.</li>
</ul>
<h2>Experiments</h2>

<h3>Federated Learning</h3>

<p>
  Experiments are conducted on the pre-processed datasets EMNIST and CIFAR-100 with a federated learning setup
  involving 100 clients.
</p>

<h3>Fairness</h3>

<p>
  We measure fairness through demographic parity, ensuring that prediction rates are similar across the different classes.
  We use a regularization term to achieve this and measure the variance on the mean of predictions for each class.
</p>

<h3>Robustness</h3>

<p>
  For robustness, we employ randomized smoothing and run multiple versions for each sample. We then measure
  robustness by comparing the accuracy on clean and noisy test data.
</p>

<h3>Privacy</h3>

<p>
  Privacy is maintained by regularizing the model to ensure it doesn't become too tailored to specific data points.
  We use the <strong>DpAdamGaussianOptimizer</strong> and measure privacy leakage using epsilon.
</p>

<h2>Contributing</h2>

<p>
  Feel free to fork this project and submit pull requests or issues. All contributions are welcome!
</p>

<h2>License</h2>

<p>
  This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.
</p>

</body>

</html>
