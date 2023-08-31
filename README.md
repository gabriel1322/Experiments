<!DOCTYPE html>
<html>

<body>

<h1>FRAP: Capturing the Fairness/Robustness/Accuracy/Privacy Trade-Off in Machine Learning</h1>

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
Before you can run this project, make sure you have the following dependencies installed:

- Python 3.6+ 
- TensorFlow 2.0+ (`pip install tensorflow`)
- TensorFlow Privacy (`pip install tensorflow_privacy`)
- TensorFlow Federated (`pip install tensorflow_federated`)

**Important Note:** TensorFlow Federated supports the following operating systems: Linux and macOS. Please ensure you're running this project on a compatible operating system.

### Installing Dependencies

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
python train_mnist.py --NUM_CLIENTS [NUM_CLIENTS] --BATCH_SIZE [BATCH_SIZE] --EPOCHS [EPOCHS] --f_param [f_param] --r_param [r_param] --l2_norm_clip [l2_norm_clip] --noise_multiplier [noise_multiplier] --number_versions [number_versions] --noise_scale [noise_scale] --num_microbatches [num_microbatches] --learning_rate [learning_rate]
``` 


<h2>Experiments</h2>

<h3>Fairness</h3>

<p>
  We measure fairness through demographic parity, ensuring that prediction rates are similar across the different classes.
  We use a regularization term to achieve this and measure the variance on the mean of predictions for each class.
</p>

<h3>Privacy</h3>

<p>
  Privacy is maintained by regularizing the model to ensure it doesn't become too tailored to specific data points.
  We use the <strong>DpAdamGaussianOptimizer</strong> and measure privacy leakage using epsilon.
</p>

<h3>Robustness</h3>

<p>
  For robustness, we employ randomized smoothing and run multiple versions for each sample. We then measure
  robustness by comparing the accuracy on clean and noisy test data.
</p>

<h3>Federated Learning</h3>

<p>
  Experiments are conducted on the pre-processed datasets EMNIST and CIFAR-100 with a federated learning setup
  involving 100 clients.
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
