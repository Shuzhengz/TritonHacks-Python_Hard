# TritonHacks Python-Hard Challenge: OpenCV

## Introduction

The engineers aboard the spaceship have created the AVHC (stands for A Very Huge Camera), a super high-resolution 
camera! The problem is the images are simply too big to be processed by the algorithms on your computer in real time.

<p style="text-align:center;"><img src="images/opencvDemo.png" alt="example" width=500 /></p>

### Challenge:

Using OpenCV’s various methods of feature detection, you will identify the features that are present in the photos that 
the AVHC took. You will then extract these features and generate separate smaller images that your computer can handle.

Through this challenge, you will learn about haar cascade and how to use some of the libraries in OpenCV's python port

## Getting Started:

### Requirements:

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

You will need OpenCV library and MatplotLib library for this challenge

We provide a `requirement.txt` file for you to quickly install all the dependencies, to install, navigate to the 
directory and enter:

`pip install -r requirements.txt`

If you would like to install them individually, run:

- For OpenCV: `pip install opencv-python==4.7.0.72`
- For MatplotLib: `pip install matplotlib==3.7.1`

