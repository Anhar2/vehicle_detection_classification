# Flask Vehicle Detection and Classification App

This is a Flask web application for object detection and classification using YOLOv5 and InceptionV3 models.

## Features

- Upload images for object detection and classification.
- Detect objects using YOLOv5 model.
- Classify objects using InceptionV3 model.

## Requirements

- Python 3.x
- Flask
- OpenCV
- TensorFlow
- YOLOv5

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/vehicle_detection_classification.git
    cd vehicle_detection_classification
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the YOLOv5 repository:
    ```sh
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt
    cd ..
    ```

5. Place the `best_inceptionv3_model_no_overfitting.h5` model in the root directory, you can download it from here: https://drive.google.com/file/d/1AShgeTsMz-uCuTRGvdT67wEAxYzYyH53/view?usp=sharing


6. Download 'yolov5x.pt' and place it inside yolov5 directory

## Configuration

- Ensure the following paths are correctly set in `app.py`:
    - `data_yaml_path = 'custom_dataset.yaml'`
    - `weights_path = 'yolov5/yolov5x.pt'`
    - `test_images_path = 'uploads/'`
    - `output_images_path = 'static/images'`

## Usage

1. Run the Flask application:
    ```sh
    python3 app.py
    ```

3. Open your web browser and go to `http://127.0.0.1:5000`.

4. Upload an image to perform object detection and classification.

## File Structure

- `app.py`: Main application file.
- `uploads/`: Directory to store uploaded images.
- `static/`: Directory for static files.
- `custom_dataset.yaml`: Configuration file for YOLOv5.
- `best_inceptionv3_model_no_overfitting.h5`: Pre-trained InceptionV3 model.

 
 
