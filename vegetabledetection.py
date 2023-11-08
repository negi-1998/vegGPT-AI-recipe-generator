import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
import openai
import streamlit as st
openai.api_key = "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"
def generate_frames(ingredients):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role":"system", "content":"Take these ingredients"+ingredients},
            {"role":"system", "content":"Generate an Indian Recipe Based on these Ingredients."}
        ]

    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
        return result

def load_yolonas_process_each_frame(video_name, stframe):
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)
    model = models.get('yolo_nas_s', num_classes= 26, checkpoint_path='weights/ckpt_best.pth')
    count = 0
    classNames = ['avocado','beans','beet','bell pepper','broccoli','brus capusta','cabbage','carrot','cayliflower','celery', 'corn', 'cucumber', 'eggplant', 'fasol','garlic','hot pepper', 'onion', 'peas', 'potato', 'pumpkin', 'rediska', 'redka', 'salad', 'squash-patisson', 'tomato', 'vegetable marrow']
    out = cv2.VideoWriter('Output9.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    class_final_names = []
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            result = list(model.predict(frame, conf=0.30))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                class_final_names.append(class_name)
                conf = math.ceil((confidence*100))/100
                label = f'{class_name}{conf}'
                print("Frame N", count, "", x1, y1,x2, y2)
                t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] -3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                #resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            stframe.image(frame, channels='BGR', use_column_width=True)
            out.write(frame)
            #cv2.imshow("Frame", resize_frame)
            #if cv2.waitKey(1) & 0xFF == ord('1'):
            #    break
        else:
            break
    lists = np.array(class_final_names)
    unique_list = np.unique(lists)
    ingredients_detected = ','.join(unique_list)
    article = generate_frames(ingredients_detected)
    st.write(article)
    out.release()
    cap.release()