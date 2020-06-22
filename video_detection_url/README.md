# video Detection with ImageAI

## How it works?
we take the url of any video and paste it in the code,now using open cv and streamlink videos are read frame by frame.now each frame is applied object detection and the output image is appended to a list.after appending all the images,each image from the list is taken and written into a video file to create a output video with detected objects.
simultaneously each detected objects image is created in a seperate folder
## How to use?
1. download this repository:

2. Create a python virtual environment and install the modules in _requirements.txt_

3.  create a model folder inside the repo download and save the yolo model file there from this [link](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5) to model folder

3. take any url link of video and put it in the code
now run the file model.ipynb
4.this file will stop running once the output video is processed it will print "finished processing" once the video is finsied processing 
now check the file video.avi which contains the detected objects.


