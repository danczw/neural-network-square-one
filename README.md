# Neural-Network-from-Scratch

##### *based on ['Neural Network from Scratch'](https://nnfs.io)*

<br>

Building a Neural Network from scratch in Python with minimal library useage. 

Follow step-by-step instructions of creating a Neural Network from scratch in */nnfs_steps*.
Important concepts can be found in */concepts*.

---

## Setup

Dependencies are listed in *environment.yml*

* run `conda env create -f environment.yml`
* activate env via `conda activate <env name>`
* run `pip install nnfs`

---

## Neural Network from Scratch

Each step builds on previous step and adds a new concept:

01. [Single layer calculation of a Neural Network (NN)](nnfs_steps\01_nnfs_single-layer01.py)
    * first single layer
02. [Single layer calculation of a NN](nnfs_steps\02_nnfs_single-layer02.py)
    * added numpy
03. [Single layer calculation of a NN](nnfs_steps\03_nnfs_single-layer03.py)
    * added batch input
04. [Multi layer calculation of a NN](nnfs_steps\04_nnfs_multi-layer01.py)
    * added further layer
05. [Multi layer calculation of a NN](nnfs_steps\05_nnfs_multi-layer02.py)
    * added OOP
06. [Multi layer calculation of a NN](nnfs_steps\06_nnfs_activation01.py)
    * added activation function for first layer
07. [Multi layer calculation of a NN](nnfs_steps\07_nnfs_activation02.py)
    * added activation function for output layer (softmax)
08. [Multi layer calculation of a NN](nnfs_steps\08_nnfs_loss.py)
    * added loss function
09. [Random optimization of a NN](nnfs_steps\09_nnfs_random-optimization.py)
    * added random weights and bias optimization
10. ~~Backpropagation of a NN~~ << coming soon >>

---

## Concepts
The purpose of the concepts is to better understand changes in the different steps and add detailed explanations.

* [Log](concepts\01_log.py)
* [Loss](concepts\02_loss.py)
* [Accuracy](concepts\03_accuracy.py)
* [Backpropagation Single Neuron](concepts\04_backpropagation_single-neuron.py)
* [Backpropagation Single Layer](concepts\05_backpropagation_single-layer.py)
* [Backpropagation Single Layer w optimization](concepts\06_backpropagation_single-layer_optimization.py)
* [Softmax Derivative and Jacobian matrix](concepts\07_softmax_derivative.py)

---

## Assets

<br>

**max() => ReLU activation derivative**

<img src=".\assets\relu_derivative.PNG" alt="ReLU derivative" width="500"/>

<br>

**Categorical Cross-Entropy loss derivative**

<img src=".\assets\cross_categorical_derivative.PNG" alt="Categorical Cross-Entropy loss derivative" width="500"/>

> i — i-th sample in a set
>
> j — label/output index
> 
> L<sub>i</sub> — sample loss value
> 
> y — target values
>
> y-hat — predicted values

<br>

**Softmax activation derivative**

<img src=".\assets\softmax_derivative.PNG" alt="Softmax activation derivative" width="500"/>

> L — number of inputs
>
> S<sub>i,j</sub> — j-th Softmax’s output of i-th sample
>
> z — input array which is a list of input vectors (output vectors from the previous layer)
> 
> z<sub>i,j</sub> — j-th Softmax’s input of i-th sample
> 
> z<sub>i,k</sub> — k-th Softmax’s input of i-th sample


<br>

**Common Categorical Cross-entropy loss and Softmax activation derivative**

<img src=".\assets\cross_categorical_softmax_derivative.PNG" alt="Categorical Cross-Entropy loss derivative" width="500"/>

> i — i-th sample in a set
>
> j — label/output index
> 
> k — index of the target label (ground-true label)
>
> L<sub>i</sub> - sample loss value
>
> S<sub>i,j</sub> — j-th Softmax’s output of i-th sample
>
> y — target values
>
> y-hat — predicted values
>
> z — input array which is a list of input vectors (output vectors from the previous layer)
>
> z<sub>i,j</sub> — j-th Softmax’s input of i-th sample
>
> z<sub>i,k</sub> — k-th Softmax’s input of i-th sample

---

#### *based on ['Neural Network from Scratch'](https://nnfs.io)*