import cv2
from ultralytics import YOLO
import pandas as pd
import gradio as gr

model = YOLO(r"best.pt")

def detect_img(image_path):
	image = cv2.imread(image_path)
	image = cv2.resize(image, (640, 640))
	results = model(image,conf=0.5) 
	data = {}
	for result in results:
		# print(result.names)
		data_dict = result.names
		example_tensor = result.boxes.cls
		count_dict = {object_name: 0 for object_name in data_dict.values()}
		for index in example_tensor:
		    item = int(index.item())
		    object_name = data_dict.get(item)
		    if object_name:
		        count_dict[object_name] += 1
		for object_name, count in count_dict.items():
	    		data[object_name] = count

	disease_list = []
	for disease, count in data.items():
	    if count > 0:
	        disease_list.append(disease)
	result_string = ', '.join(disease_list)
	if len(result_string) > 2:
		fnl_str = result_string
	else:
		fnl_str = "There was no disease detected"

	if results:
		print(results)
		for detection in results:
			x_min, y_min, x_max, y_max = detection.boxes.xyxy[0].tolist()
			cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
			label = fnl_str
			cv2.putText(image, label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

	output_image_path = "./bound.png"
	cv2.imwrite(output_image_path, image)
	return fnl_str


def ret_str(image_path):
	print(type(image_path))
	res = detect_img(image_path)

	processed_image = "./bound.png"
	return "Disease found: " + res, processed_image


inputs = gr.Image(type="filepath", label="Upload Image") 
outputs = [gr.Textbox(label="Predicted Text"), gr.Image(label="Processed Image")] 

gr.Interface(fn=ret_str, inputs=inputs, outputs=outputs).launch()

