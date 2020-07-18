# Face recognition and identification OpenCV in Python.
---

This is a python 3.X face and eye recognition app using the OpenCV library including a trainig model.

### Dependencies
---
Numpy 
```
pip install numpy --upgrade
```
Opencv Windows
```
pip install opencv-contrib-python --upgrade
```
Pillow
```
pip install pillow --upgrade
```

### How to use it
---
1. Train the model
	- First of all, save a couple of photos of the person you want to detect into the "images" folder, there you should save those in a sub folder with the name of the person (examples in "images" folder in repo).

	- Then make sure you have at least two sub folders with photos of two different persons.

	- And finally execute the "face_trainer.py", wait until it prints "Train done!" in console, it will create "trainner.yml" file in the root of the project.

2. Detect face
	- Execute "face_detector.py"
