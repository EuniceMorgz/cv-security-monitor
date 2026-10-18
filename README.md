# Motion-Triggered Security Monitor with Multi-Object Tracking

CSC 126 Graphics and Visual Computing - Midterm Project
Name: Morgadez, Eunice A., Salazar, Kimberlee Kate V.

## 1. Description
The Motion-Triggered Security Monitor is a desktop application built with Python, 
OpenCV, and Tkinter. It processes a live webcam feed or a recorded video, finds the 
parts of each frame that are moving, draws bounding boxes around them, tracks each moving 
object from frame to frame, and shows the results in real time. It is packaged as a Windows 
program (MotionMonitor.exe) so it can run without installing Python.

## Techniques Used
-Image Operations: resize, Gaussian blur, morphological opening and dilation
-Image Segmentation: MOG2 background subtraction and thresholding
-Motion and Object Tracking: contours, bounding boxes, merged boxes, centroid tracker

## Setup
Option A - Use the ready-made app (no Python needed):

1. Open the Releases page of this repository.
2. Download MotionMonitor.exe.
3. Double-click it to run. The first start may take a few seconds.
4. If Windows says "Windows protected your PC", click More info, then Run anyway.

Option B - Run from the source code:

1. Install Python 3.10 or newer from python.org. Tick "Add python.exe to PATH" during installation.
2. Download this repository and open a terminal in its folder.
3. Create a virtual environment by typing: python -m venv venv
4. Activate it by typing: venv\Scripts\activate
5. Install the libraries by typing: pip install -r requirements.txt

## How to Run
To run from the source code, type: python app.py

Or double-click MotionMonitor.exe.

How to use the window:

1. Click Start Webcam. Stay out of the camera view for about 1 second while it learns the background.
2. When the status says "Clear - no motion", walk in front of the camera.
3. A green box, an ID label, and a blue trail follow each moving object, and the status turns red.
4. To use a video instead, click Open Video File and choose samples/test.mp4.
5. Click Reset Background if the lighting changes.
6. Click Save Screenshot to save the current picture.
7. Click Stop to stop.

## Folders
- samples - sample input video
- captures - snapshots saved automatically when motion is detected
- screenshots - screenshots of the program output