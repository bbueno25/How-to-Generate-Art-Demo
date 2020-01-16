# How To Generate Art Demo

## Overview

We're going to re-purpose the pre-trained VGG16 convolutional network that won the ImageNet competition in 2014 to transfer the style of a given image to another.

- [video](https://youtu.be/Oex0eWoU7AQ)  

## Usage

`pip install -r requirements.txt` 

Create a file called ~/.keras/keras.json and make sure it looks like the following:

   ````
   {
       "image_dim_ordering": "tf",
       "epsilon": 1e-07,
       "floatx": "float32",
       "backend": "tensorflow"
   }
   ````

`jupyter notebook`

`python demo.py`

## Contributors

- [Harish Narayanan](https://github.com/hnarayanan)
- [Siraj Raval](https://github.com/llSourcell)
- [B. Bueno](https://github.com/bbueno25)
