# Face Recognition Based Smart Lock

This project implements a face recognition-based smart lock system using Python. The system can identify authorized personnel and record unauthorized access attempts with corresponding images and timestamps in an Excel file.

## Features

- Face detection and recognition using OpenCV.
- Recording unauthorized access attempts in an Excel file with images and timestamps.
- Easy-to-use interface with real-time face recognition.

## Requirements

- Python 3.8
- OpenCV
- Openpyxl

## Installation

1. Clone the repository:
    ```sh
    https://github.com/Krushnansh22/Face_Recognition_Based_SmartLock.git
    cd Face_Recognition_Based_SmartLock
    ```

2. Install the required packages:
    ```sh
    pip install opencv-contrib-python openpyxl 
    ```

## Usage

### Data Collection & Training

1. To collect data for training the face recognition model, run the `DataCollection.py` script. This script will capture images from your webcam and save them in the specified directory.

2. After collecting the data, the script trains the face recognition model. This script will read the collected images, train a face recognizer, and save the model and label dictionary.
  
   ```sh
    python DataCollection.py
    ```

### Adding Unauthorized Unlock Attempts to Excel

1. The `AddToExcel.py` script contains the `unauthorized_unlock` function which saves the unauthorized attempt details to an Excel file. This function is called automatically during the prediction phase if an unauthorized access attempt is detected.

### Predicting Faces and Unlocking

1. To run the face recognition and smart lock system, execute the `Predict.py` script. This script will use the trained face recognition model to identify faces from the webcam feed and display real-time results.

    ```sh
    python Predict.py
    ```

2. If an unauthorized person attempts to unlock, their image and the attempt details will be saved in the `unauthorized.xlsx` file.

## Files

- `DataCollection.py`: Script for collecting training data and training the model.
- `AddToExcel.py`: Script for recording unauthorized access attempts.
- `Predict.py`: Script for running the face recognition and smart lock system.

## Model and Labels

- Ensure you have the trained face recognition model (`face_recognizer_model.xml`) and the label dictionary (`label_dict.pkl`) in the same directory as the scripts before running the prediction script.

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenCV for computer vision functionalities.
- Openpyxl for handling Excel files.

