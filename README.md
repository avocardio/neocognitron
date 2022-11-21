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

Usage:
```
python test.py
```

## Requirements
* numpy
* tensorflow
* matplotlib
* opencv-python