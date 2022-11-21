# neocognitron
Testing the neocognitron (1979) on MNIST digits (1998).

<img src="image.png" alt="example" width="400"/>

Using the neocognitron implementation from: https://github.com/altugkarakurt/NeuralHDR


## What does it do?
It fowards 10 images (0-9) of each MNIST digit, and then finally predicts the number 5 image.
```
for i in range(10):
	neocognitron.estimate(mnist[i])

neocognitron.estimate(mnist[5])
```
The output is an array of 10 values, where the index of the highest value is the predicted digit.
```
[0.6679867, 0.6678071, 0.6658286, 0.6665703, 0.6675671, 0.6624054, 0.6684094, 0.665012, 0.6660348, 0.6659308]
```

## Usage:

Tune hyperparameters in `test.py` 
```
N = Number of digits for forwarding (has to be 10 or more, divisible by 10)
NUMBER_TO_PREDICT = Number to predict (0 - 9)
```
And then:
```
python test.py
```

## Requirements
* numpy
* tensorflow
* matplotlib
* opencv-python
* tqdm
