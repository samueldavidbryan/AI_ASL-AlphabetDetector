# American Sign Language Translator Using Mediapipe

**To run the demo:**

```
python3 app.py

```

## Project File Structure 


	- Used to classify key points for handshapes.

	- Runs the demo

**Directories:**

- *model / keypoint_classifider:* Contains classification for handshapes

	- *keypoint\_classifier\_label.csv:* Labels of each handshape 

	- *keypoint_classifier.tflite:* TF Lite
	- *keypoint_classifier.hdf5:* Model

- *model / point\_history\_classifier:* Contains classification for signs involving hand movement

	- *point\_history\_classifier\_label.csv:* Labels of the signs
	- *point\_history.csv:* Instances of signs
	- *point\_history\_classifier.py:*