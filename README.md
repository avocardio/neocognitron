# neocognitron
Testing the neocognitron (1979) on MNIST digits (1998).

<img src="image.png" alt="example" width="400"/>

Using the neocognitron implementation from: https://github.com/altugkarakurt/NeuralHDR


## Usage
It fowards 10 images (0-9) of each MNIST digit, and then finally predicts the number 5 image.
```
for i in range(10):
	neocognitron.estimate(mnist[i])

neocognitron.estimate(mnist[5])
```
The output is an array of 10 values, where the index of the highest value is the predicted digit.
```
[array([[0.67101553]]), array([[0.66803563]]), array([[0.66949816]]), array([[0.66904382]]), array([[0.66371704]]), array([[0.65974766]]), array([[0.66211057]]), array([[0.66890992]]), array([[0.67033799]]), array([[0.66387558]])]
```

Usage:
```
python test.py
```

## Requirements
* numpy
* tensorflow
* matplotlib
* opencv-python