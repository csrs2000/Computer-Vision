from flask import Flask, Response, jsonify
app = Flask(__name__)
import os
from imageai.Detection import ObjectDetection
import time
import json

execution_path = os.getcwd()
st = time.time()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path,"yolo.h5"))
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(os.path.join(execution_path, "model", "yolo-tiny.h5"))
detector.loadModel()
# detector.loadModel(detection_speed="fastest")
print(f'Init Timer: {time.time()-st}')

@app.route('/detect/<pic_name>')
def boat_detection(pic_name):
    st = time.time()
    results = getDetections(pic_name)
    print(f'Sum Timer: {time.time()-st}')

    msg = {}
    for i, result in enumerate(results, 1):
        result['percentage_probability'] = float(result['percentage_probability'])
        result['box_points'] = list(result['box_points'])
        for index in range(len(result['box_points'])):
            result['box_points'][index] = int(result['box_points'][index])
        result['box_points'] = tuple(result['box_points'])
        msg[str(i)] = json.dumps(result)
    return jsonify(msg)


def getDetections(file_name):
    start = time.time()

    image_folder = os.path.join(execution_path)
    output_folder = os.path.join(execution_path)

    st1 = time.time()
    image_file = os.path.join(image_folder, file_name)
    new_image_file = os.path.join(output_folder, file_name)
    print(image_file, "-->", new_image_file)
    if not os.path.exists(image_file):
        print("not exist.")
        return

    # global detector
    custom_objects = detector.CustomObjects(person=True)
    detections = detector.detectCustomObjectsFromImage(
        custom_objects=custom_objects,
        input_image=image_file,
        output_image_path=new_image_file,
        minimum_percentage_probability=30)
    print(f'[Info]识别到 boat{len(detections)}艘')
    for eachObject in detections:
        print(eachObject.items())

    end = time.time()
    print(f'Excute Timer: {end-st1}')
    print ("耗时: ",end-start)
    return detections

if __name__ == '__main__':
    app.run(threaded=False) # ban the Threaded mode