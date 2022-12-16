import uvicorn
from fastapi import FastAPI, Form
import onnxruntime
import cv2
import os
import sys
import requests
import json
import time
import numpy as np
import base64
from datetime import datetime
from model import inference
import torch
from torchvision.ops import box_iou
import traceback


class TrackingPipeline:
    def __init__(
        self,
        consider_iou = True,
        iou_threshold = 0.2,
    ):
        self.consider_iou = consider_iou
        self.iou_threshold = iou_threshold

        self.previous_boxes = None
        self.previous_iou_ids = None

        self.id_to_box = {}

    def perform_inference(self, boxes):
        if self.previous_boxes is None:
            self.previous_iou_ids = np.arange(len(boxes))
            self.previous_boxes = torch.from_numpy(boxes).reshape(-1, 4)

        self.boxes = torch.from_numpy(boxes).reshape(-1, 4)

        ious = box_iou(self.boxes, self.previous_boxes)
        max_ious, max_iou_indices = ious.max(1)

        self.iou_ids = np.empty(len(self.boxes)).astype(int)
        iou_condition_is_true = max_ious >= self.iou_threshold
        self.iou_ids[iou_condition_is_true] = self.previous_iou_ids[max_iou_indices[iou_condition_is_true]]
        
        iou_condition_is_false = max_ious < self.iou_threshold
        new_boxes = self.boxes[iou_condition_is_false]
        new_ids = np.arange(1, len(new_boxes) + 1) + self.previous_iou_ids.max()
        self.iou_ids[iou_condition_is_false] = new_ids

        return self.iou_ids
    
    def update_previous_boxes(self):
        for iou_id, box in zip(self.iou_ids, self.boxes.numpy()):
            self.id_to_box[iou_id] = box.reshape(2, 2)

        self.previous_boxes = self.boxes
        self.previous_iou_ids = self.iou_ids


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


app = FastAPI(debug = False)

# session = onnxruntime.InferenceSession(resource_path('m1.onnx'), providers = ['CPUExecutionProvider'])  #  -> to compile exe
session = onnxruntime.InferenceSession(
    resource_path('assets/m1.onnx'),
    providers = ['CPUExecutionProvider'],
)
last_results = None
last_ids = None
last_frame_idx = None
actual_date = None
tracking_pipeline = TrackingPipeline(iou_threshold = 0.15)
global_frame_idx_to_results = {}

def verify_license():
    global actual_date

    actual_day = int(actual_date.split('-')[2])
    actual_month = int(actual_date.split('-')[1])
    actual_year = int(actual_date.split('-')[0])

    pc_datetime = datetime.now().date()

    pc_day = pc_datetime.day
    pc_month = pc_datetime.month
    pc_year = pc_datetime.year

    if not (pc_month == actual_month and pc_year == actual_year and pc_day == actual_day):
        get_correct_date_time()
        if not (pc_month == actual_month and pc_year == actual_year and pc_day == actual_day):
            return '\nLicense verification failed, you have changed the time of your system :|\n'
    
    if pc_month >= 2 and pc_year >= 2023:
        get_correct_date_time()

        return '\nTime trial is over! Contact Skylarklabs.ai for further discussion\n'

@app.post('/inference/')
def inference_view(
    file: str = Form(default = None),
    imgsz: int = Form(default = 736),
    conf: float = Form(default = 0.2),
    iou: float = Form(default = 0.4),
    frame_idx: str = Form(default = '0'),
    sampling_step: str = Form(default = '5'),
):
    global last_results, last_ids, last_frame_idx, tracking_pipeline, global_frame_idx_to_results

    frame_idx = int(frame_idx)
    sampling_step = int(sampling_step)
    
    try:
        if frame_idx % sampling_step == 0:
            cv2_encoded = np.frombuffer(base64.decodebytes(bytes(file, 'utf-8')), dtype = "uint8")
            img = cv2.imdecode(cv2_encoded, 1)

            current_boxes, current_labels = inference(session, img, imgsz, conf, iou)
        
            if len(current_boxes) >= 2:

                current_boxes_np = np.array(current_boxes)
                matched_ids = tracking_pipeline.perform_inference(current_boxes_np).tolist()

                if last_frame_idx is None:
                    last_frame_idx = 1
                    
                num_interpolations = frame_idx - last_frame_idx

                if num_interpolations - 1 > 0 and last_frame_idx != 1:
                    interpolated_previous_boxes = np.empty((num_interpolations, len(matched_ids), 2, 2)).astype(int)
                    last_frame_idx = frame_idx

                    for box_idx, (iou_id, current_box) in enumerate(zip(matched_ids, current_boxes_np)):
                        if iou_id in tracking_pipeline.previous_iou_ids:
                            previous_box = tracking_pipeline.id_to_box[iou_id]
                            inters = []

                            for alpha in np.arange(1.0, 2.0 + 1.0 / ((num_interpolations - 1)), 1.0 / ((num_interpolations - 1))):
                                inters.append(current_box * alpha + previous_box * (1 - alpha))
                        else:
                            inters = [current_box] * (num_interpolations)

                        inters = np.array(inters).astype(int)
                        interpolated_previous_boxes[:, box_idx, :, :] = inters
                else:
                    last_frame_idx = frame_idx
                    interpolated_previous_boxes = np.array([])

                tracking_pipeline.update_previous_boxes()

                for idx, boxes in enumerate(interpolated_previous_boxes):
                    global_frame_idx_to_results[frame_idx + idx] = boxes
                
        return global_frame_idx_to_results.pop(frame_idx).tolist()
    
    except Exception as e:
        print('\n\nAn error has occured during inference:\n')
        print(traceback.format_exc())
        print('\n', 'Report this to SkylarkLabs.ai and send the sample video you are testing this exe on.')

    return []


@app.get('/')
def home():
    return {"message": "Read the provided README.md file for usage guide."}


@app.get('/inference/')
def inference_view():
    return {"message": "Read the provided README.md file for usage guide."}


def get_correct_date_time():
    global actual_date
    trial = 0

    while trial < 5:
        try:
            respones = requests.get("http://worldtimeapi.org/api/timezone/Asia/Kolkata")
            break
        except:
            print('No internet connection found! Trying again...')
            trial += 1
    
    if trial == 5:
        print('\nNo internet connection found! ')
        time.sleep(5)
        print('Terminating...')
        sys.exit(0)

    if respones.status_code == 200:
        respones = json.loads(respones.text)
        actual_date = respones['datetime'][:10]
    else:
        print('\nSome issue has occured with license verification, contact Skylarklabs.ai for help!\n')
        time.sleep(5)
        print('Terminating...')
        sys.exit(0)


get_correct_date_time()

has_license_issues = verify_license()

if has_license_issues:
    print(has_license_issues)
    exit(0)
    
print('\n\n[=============== Starting Server ===============]\n\n')
uvicorn.run(app, host = "0.0.0.0", port = 9095)