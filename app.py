from flask import Flask, request, render_template, redirect
import shutil, glob
import gc
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['STATIC_FOLDER'] = 'static/'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #no need for GPU here

data_yaml_path = 'custom_dataset.yaml'    
weights_path = 'yolov5/yolov5x.pt'   
test_images_path = 'uploads/'

class_labels = ['bus',  'car',  'truck']

#Just RUNNING The detecty.py in yolov5 with the extra large model yolov5x.pt to localize the objects in the image
def run_inference(data_yaml, weights, source, img_size=640, conf_thres=0.25, project='uploads', name='yolo'):
    os.system(f'python3 yolov5/detect.py --weights {weights} --img {img_size} --data {data_yaml} --conf {conf_thres} --source {source} --classes 2 5 7 --save-txt --save-conf --project {project} --name {name}')
 
#Resizing the input, and making classification based on the weights of the best model I could get
def classify_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    classification_model = load_model('best_inceptionv3_model_no_overfitting.h5') 
    preds = classification_model.predict(image)
    del classification_model
    gc.collect()
    return preds

#Parsing the output of the yolo model to get the bounding boxes
def parse_yolo_output(labels_path='uploads/yolo/labels'):
    bboxes = []
    for label_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_file), 'r') as file:
            for line in file:
                _, x, y, w, h,_ = map(float, line.strip().split())
                bboxes.append((x, y, w, h))
    yolo_folders = glob.glob('uploads/yolo*')
    for folder in yolo_folders:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    return bboxes

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}
#Cropping the image based on the bounding boxes to send this part specifically to be classified
#Since it was the way the model was trained
def crop_image(image, bboxes):
    cropped_images = []
    height, width, _ = image.shape
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)
    return cropped_images

#Just drawing the bounding boxes and the classified labels on the original image
def draw_boxes_and_labels(image, bboxes, predictions):
    height, width, _ = image.shape

    for bbox, prediction in zip(bboxes, predictions):
        x, y, w, h = bbox
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        class_index = np.argmax(prediction)
        label = class_labels[class_index]
        text = f'{label}'        
        # Draw the label
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            run_inference(data_yaml_path, weights_path, file_path)
            
            bboxes = parse_yolo_output()
            
            image = cv2.imread(file_path)
            
            # Crop the image based on the bounding box(es)
            cropped_images = crop_image(image, bboxes)            
            predictions = [classify_image(cropped_image) for cropped_image in cropped_images]

            annotated_image = draw_boxes_and_labels(image, bboxes, predictions)
            annotated_image_path = os.path.join(app.config['STATIC_FOLDER'], 'images/annotated_' + filename)
            cv2.imwrite(annotated_image_path, annotated_image)
            print(annotated_image_path)

            
            return render_template('result.html', image_path=annotated_image_path)
    return render_template('start.html')

if __name__ == '__main__':
    app.run(debug=True)