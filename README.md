# TritonHacks-Python_Hard-OpenCV
TritonHacks: Python Hard openCV and NumPy image manipulation challenge

## Background

Earth is invaded by aliens; therefore, you (the programmer) have to evacuate earth on a spaceship. The spaceship will have a camera that will take pictures 
once in a while when you are in space. For each image taken, you have to perform a series of tasks to decode the image to see if there is an alien among us.

## Challenge

- OpenCV

  - **Background story (context)**: The engineers aboard the spaceship have created the AVHC (stands for A Very Huge Camera), a super high-resolution camera! 
  The problem is the images are simply too big to be processed by the algorithms on your computer in real time.

  - **Challenge**: Using OpenCV’s various methods of feature detection, you will identify important features that are present in the photos that the AVHC took. 
  You will then extract these features and generate smaller images that your computer can handle. Each of your generated images cannot exceed 3MP in resolution.

- NumPy (+ maybe scripy)

  - **Background story (context)**: Good job, you have spoiled many evil plans from the aliens, but they will not stand there and do nothing. 
  Recently, the angry aliens have started using pictures to trick our algorithms! To combat this and ensure the human’s safety, 
  you will have to distinguish between pictures of humans (they are aliens in disguise!) and real humans passing in front of the security cameras.

  - **Challenge**: Using NumPy’s powerful image manipulation tools, you will extract normal maps from pictures the security cameras took in order to 
  obtain depth information to distinguish between pictures of humans (the aliens are using them as face-covers) and real humans. 
  You might also need to use NumPy’s segmentation tools to remove backgrounds to get the best result.
  
