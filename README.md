# TritonHacks 2023 - Python Hard Challenge

TritonHacks: Python Hard. An OpenCV and NumPy image manipulation challenge

[Link to event Devpost](https://tritonhacks23.devpost.com/), [Link to the published hackathon](https://tritonhacks.notion.site/Journey-Through-Space-with-Machine-Learning-655a11aea9784c0eb25b393b1991596a),
**event has concluded**

Note that the published version has been simplified from the version in this repo

## Background

Earth is invaded by aliens; therefore, you (the programmer) have to evacuate earth on a spaceship. The spaceship will have a camera that will take pictures 
once in a while when you are in space. For each image taken, you have to perform a series of tasks to decode the image to see if there is an alien among us.

## Challenge

- OpenCV

  - **Background story (context)**: The engineers aboard the spaceship have created the AVHC (stands for A Very Huge Camera), a super high-resolution camera! 
  The problem is the images are simply too big to be processed by the algorithms on your computer in real time.

  - **Challenge**: Using OpenCV’s various methods of feature detection, you will identify the features that are present in the photos that the AVHC took. 
  You will then extract these features and generate separate smaller images that your computer can handle.

- NumPy (+ maybe scipy)

  - **Background story (context)**: Good job, you have spoiled many evil plans from the aliens, but they will not stand there and do nothing. 
  Recently, the angry aliens have started using pictures to trick our algorithms! To combat this and ensure the human’s safety, 
  you will have to distinguish between pictures of humans (they are aliens in disguise!) and real humans passing in front of the security cameras.

  - **Challenge**: Using NumPy’s powerful image manipulation tools, you will seperate objects with the backgroundand extract the normal map from pictures the security
  cameras took in order to obtain depth information to distinguish between pictures of humans (the aliens are using them as face-covers) and real humans. 
  You might also need to use NumPy’s segmentation tools and some OpenCV libraries to get the best result.
  
## Requirements

Requirement packages and dependencies are listed in `requirements.txt`

To insall all requirements, navigate into the repository directory and enter `pip install -r requirements.txt` in the terminal

#### Note: Mediapipe does not support Python 3.11 at the moment, please use Python 3.10 instead
