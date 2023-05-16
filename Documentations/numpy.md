# TritonHacks Python-Hard Challenge: NumPy

Earth is invaded by aliens; therefore, you (the programmer) have to evacuate earth on a spaceship. The spaceship will 
have a camera that will take pictures once in a while when you are in space. For each image taken, you have to perform 
a series of tasks to decode the image to see if there is an alien among us.

## Introduction

Good job, you have spoiled many evil plans from the aliens, but they will not stand there and do nothing. 
Recently, the angry aliens have started using pictures to trick our algorithms! To combat this and ensure the human’s 
safety, you will have to distinguish between pictures of humans (they are aliens in disguise!) and real humans passing 
in front of the security cameras.

### Challenge

Using NumPy’s powerful image manipulation tools, you will separate objects with the background and extract the normal 
map from pictures the security cameras took in order to obtain depth information for our computers to distinguish 
between pictures of humans (the aliens are using them as face-covers) and real humans. You also need to use NumPy’s 
segmentation tools and some OpenCV libraries to get the best result.
  
## Requirements:

You need python to do this, hopefully you have that installed, if not, visit python's website at 
[https://www.python.org](https://www.python.org)

<h4>Make sure to have pip installed with python, pip is the package manager for python and is very useful for managing 
your dependencies</h4>

To install pip, run:

- On Linux: `python -m ensurepip --upgrade`
- On Windows: `py -m ensurepip --upgrade`
- On MacOS: `python -m ensurepip --upgrade`

Make sure to keep pip updated, to do this, run: 
`pip install --upgrade pip`

In your starter kit there should be a few demo images, make sure they are in the same 
directory as your code