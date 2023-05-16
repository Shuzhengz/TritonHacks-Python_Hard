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

#### Important: Mediapipe currently only supports up to Python 3.10, please do not use Python 3.11 for this challenge

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

### Dependencies

You will need NumPy, Mediapipe, OpenCV, and MatplotLib for this challenge

We provide a `requirement.txt` file for you to quickly install all the dependencies, to install, navigate to the 
directory and enter:

`pip install -r requirements.txt`

If you would like to install them individually, run:

- For NumPy: `pip install numpy==1.24.2`
- For Mediapipe: `pip install mediapipe==0.9.3.0`
- For OpenCV: `pip install opencv-python==4.7.0.72`
- For MatplotLib: `pip install matplotlib==3.7.1`

<h6> Note: OpenCV's Python port and Matplotlib currently do not support devices with Apple Silicon. </h6>

## Part 1: Setting Up

### Dependencies

Create a file `<give it a name>.py` in your project directory, make sure that the images that you use are in the same 
directory as your code