import cv2
import json
import base64
import requests
import time
from collections import deque

def api_preprocess(frame):
    encoded = base64.b64encode(cv2.imencode('.jpg', frame)[1])
    return encoded

frame_idx = 0
previous_frames = []
latest_results = None
inference_tock = 0
cum_inference_tock = 0.0

def accumulate_global_output_frames():
	global previous_frames, frame_idx, polling_rate, global_output_frames, boxes, latest_results, inference_tock

	try:
		while True:
			has_frame, frame = cap.read()

			if has_frame:
				frame = cv2.resize(frame, (736, 736)) # do not remove this line
				
				inference_tick = time.time()
				encoded_frame = api_preprocess(frame)
				response = requests.post(
					'http://127.0.0.1:9095/inference/',
					{
						'file': encoded_frame,
						'frame_idx': str(frame_idx),
						'sampling_step': str(5)
					}
				)
				inference_tock = time.time() - inference_tick

				if response.status_code == 200:
					results = json.loads(response.text)

					global_output_frames.append((frame, results))

				frame_idx += 1

			else:
				break
	except KeyboardInterrupt:
		exit(0)
	

def visualize_live_playback():
	global global_output_frames, fps, cum_inference_tock
	frame_index_20 = 0
	tic_20 = None

	tic_cum = None
	frame_index_cum = 0
	
	fps_20 = None

	while True:
		if len(global_output_frames):
			if frame_index_20 % 20 == 0:
				tic_20 = time.time()
				frame_index_20 = 0

			if tic_cum is None:
				tic_cum = time.time()

			frame_index_20 += 1
			frame_index_cum += 1
			output_frame, boxes = global_output_frames.popleft()

			for box in boxes:
				top_left, bottom_right = box
				cv2.rectangle(output_frame, top_left, bottom_right, [0, 0, 255], 2)

			if frame_index_20 == 20:
				fps_20 = f'Avg Playback Fps({((frame_index_20) / (time.time() - tic_20 + 1e-8)):.3f})'

			cv2.putText(
				img = output_frame,
				text = fps_20,
				org = (10, 40),
				fontScale = 1,
				fontFace = 0,
				color = [0, 200, 0],
				thickness = 4,
			)

			cv2.putText(
				img = output_frame,
				text = f'Current Playback Fps({((frame_index_cum) / (time.time() - tic_cum + 1e-8)):.3f})',
				org = (10, 85),
				fontScale = 1,
				fontFace = 0,
				color = [0, 200, 0],
				thickness = 4,
			)
			cum_inference_tock = 0.8 * cum_inference_tock + 0.2 * inference_tock

			cv2.putText(
				img = output_frame,
				text = f'Inference Fps({1 / cum_inference_tock:.3f})',
				org = (10, 125),
				fontScale = 1,
				fontFace = 0,
				color = [0, 200, 0],
				thickness = 4,
			)

			cv2.imshow("Outputs", cv2.resize(output_frame, (1280, 720)))

		cv2.waitKey(5)

if __name__ == '__main__':
	from threading import Thread

	cap = cv2.VideoCapture('test2.mp4')
	fps = int(cap.get(cv2.CAP_PROP_FPS))

	sampling_fps = 5
	polling_rate = int(fps / sampling_fps)
	global_output_frames = deque()

	t1 = Thread(target = accumulate_global_output_frames)
	t2 = Thread(target = visualize_live_playback)
	
	t1.start()
	t2.start()

	t1.join()
	t2.join()