# American Sign Language Translator Using Mediapipe

**To run the demo:**

```
python3 app.py

```

## Project File Structure 
- keypoint\_classification.ipynb

	- Used to classify key points for handshapes.- point\_history\_classification.ipynb	- Used to finalize key points for signs involving motion.- app.py

	- Runs the demo- file_input.py- webcam\_map\_body.py

**Directories:**

- *model / keypoint_classifider:* Contains classification for handshapes

	- *keypoint\_classifier\_label.csv:* Labels of each handshape 	- *keypoint.csv:* Instances of handshapes

	- *keypoint_classifier.tflite:* TF Lite	
	- *keypoint_classifier.hdf5:* Model	- *keypoint\_classifier.py:*	- *keypoint\_creator.py:* Uses images to create instances of hanshapes. Input is the training set. Output is keypoints.csv in this directory. 

- *model / point\_history\_classifier:* Contains classification for signs involving hand movement

	- *point\_history\_classifier\_label.csv:* Labels of the signs	
	- *point\_history.csv:* Instances of signs	
	- *point\_history\_classifier.py:*