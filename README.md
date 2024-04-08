# Face Recognition Attendance System

This is a Python script for a face recognition attendance system using OpenCV, NumPy, and face_recognition libraries.

## Features

- Recognizes faces in real-time using a webcam.
- Matches detected faces with pre-trained images for attendance marking.
- Marks attendance in a CSV file with timestamps.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python (version 3.6 or higher)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- face_recognition (`pip install face_recognition`)

## Usage

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/IshitaChoudhuri/AttendancePortal.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd AttendancePortal
    ```

3. **Install Requirements:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare Training Images:**

    - Create a directory named `Training_images`.
    - Organize training images in subdirectories with the name of the person.

5. **Run the Script:**

    ```bash
    python main.py
    ```

6. **Interact with the System:**

    - The webcam will open, and faces will be detected in real-time.
    - When a recognized face is detected, it will be highlighted with a rectangle, and the person's name will be displayed.
    - Press 'q' to quit the webcam.

7. **Check Attendance:**

    - The attendance will be marked in a file named `Attendance.csv` in the project directory.

## Author

- Ishita Choudhuri (https://github.com/IshitaChoudhuri)

