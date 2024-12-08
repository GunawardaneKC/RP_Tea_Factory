from flask import Flask, request, jsonify
import cv2
import ultralytics
import numpy as np

model = ultralytics.YOLO("eye_new.pt")

app = Flask(__name__)

def detect_plant(image):
    try:
        results = model(image)
        
        if results:
            json_response = {"objects": []} 
            for detection in results:
                try:
                    class_index = detection.boxes.cls[0].item()
                    if class_index in detection.names:
                        # json_response['class'] = detection.names[class_index]
                        json_response["objects"].append(detection.names[class_index])
                    else:
                        json_response['class'] = 'Unknown Class'
            
                    
                except IndexError:
                    json_response['error'] = 'Enter a valid Image'
            return json_response
    except Exception as e:
        return {'error': str(e)}
    
def detect_all_plants(image):
    try:
        results = model(image)

        if results:
            json_response = {"objects": []}  # List to store all detected objects
            seen_objects = set()
            for detection in results:
                try:
                    class_indices = detection.boxes.cls
                    for class_index in class_indices:
                        class_name = detection.names[class_index.item()]

                        if class_name not in seen_objects:
                            json_response["objects"].append(class_name)
                            seen_objects.add(class_name)
                except IndexError:
                    json_response['error'] = 'Enter a valid Image'

            return json_response
    except Exception as e:
        return {'error': str(e)}


@app.route("/detect", methods=["POST"])
def detect_patient_route():

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files["image"]

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    json_response = detect_plant(image)

    return jsonify(json_response)

if __name__ == "__main__":
    app.run(debug=True)
