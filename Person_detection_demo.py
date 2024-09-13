from ultralytics import YOLO
import cv2

device = "cpu"

def draw_bounding_boxes_with_confidence(frame, boxes, confidences, color=(0, 255, 0), thickness=2, font_scale=0.5):

    for i, (box, confidence) in enumerate(zip(boxes, confidences)):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box)

        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Prepare the label with the confidence score
        label = f"{confidence:.2f}"

        # Calculate the position for the label (above the top-left corner of the bounding box)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # Draw the label background rectangle
        frame = cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Put the text (confidence score) on the frame
        frame = cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return frame

model = YOLO('yolov8x.pt')

#geting names from classes
dict_classes = model.model.names

im_path=r"Detection/Persons.png"
img=cv2.imread(im_path)

class_IDS = [0]
conf_level = 0.8
scale_percent = 100

y_hat = model.predict(img, conf = conf_level, classes = class_IDS, device = device, verbose = False)

boxes = y_hat[0].boxes.xyxy.cpu().numpy()
conf = y_hat[0].boxes.conf.cpu().numpy()
classes = y_hat[0].boxes.cls.cpu().numpy()

det_img=draw_bounding_boxes_with_confidence(img, boxes, conf)

cv2.imshow("img",det_img)
cv2.waitKey()

cv2.imwrite("Detection/output.png",det_img)