# Face Recognition Attendance System

A real-time attendance management system that leverages facial recognition and computer vision to automate attendance tracking. The application identifies registered individuals through a webcam feed, records attendance with timestamps, and detects unrecognized faces for further review.

## Features

* Real-time face detection and recognition
* Automated attendance logging with timestamps
* Single-entry attendance marking to prevent duplicates
* Unknown face detection and image capture
* Optimized recognition workflow with cooldown intervals
* Simple and user-friendly interface

## Technology Stack

* Python
* OpenCV
* face_recognition
* NumPy

## Project Structure

```
Face-Recognition-Attendance-System/
│
├── main.py
├── requirements.txt
├── screenshots/
├── images/
├── unknown_faces/
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Add reference images of registered individuals to the `images` directory.
2. Run the application:

```bash
python main.py
```

3. The system will:

   * Detect and recognize registered individuals.
   * Record attendance automatically.
   * Save unrecognized faces for review.

## Applications

* Academic institutions
* Corporate attendance systems
* Event participant verification
* Access monitoring solutions

## Future Enhancements

* Database integration
* Cloud-based attendance storage
* Multi-camera support
* Web dashboard and analytics
* Role-based authentication

## Author

Anvesh 

