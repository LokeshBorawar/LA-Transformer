import cv2
from ultralytics import YOLO
import os
import numpy as np
import torch
from torchvision import transforms
import timm
from LATransformer.model import ClassBlock, LATransformer, LATransformerTest
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import random

device = "cpu"

det_model = YOLO('yolov8x.pt')

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base= vit_base.to(device)
# Create La-Transformer
feature_model = LATransformerTest(vit_base, lmbd=8).to(device)
# Load LA-Transformer
save_path = os.path.join('./model','net_best.pth')
feature_model.load_state_dict(torch.load(save_path,map_location=device), strict=False)
feature_model.eval()

#geting names from classes
dict_classes = det_model.model.names
class_IDS = [0]
conf_level = 0.6
threshold = 0.5

id_to_color = {}
def generate_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def draw_bounding_boxes_with_ids(frame, boxes, ids, thickness=2, font_scale=0.5):
    global id_to_color

    for i, (box, person_id) in enumerate(zip(boxes, ids)):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box)

        if person_id not in id_to_color:
            id_to_color[person_id] = generate_random_color()
        color = id_to_color[person_id]

        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Prepare the label with the confidence score
        label = f"ID: {person_id}"

        # Calculate the position for the label (above the top-left corner of the bounding box)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # Draw the label background rectangle
        frame = cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Put the text (confidence score) on the frame
        frame = cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    return frame

def crop_and_resize_persons(frame, boxes, target_size=(224, 224)):

    cropped_images = []
    
    # Iterate over each bounding box
    for (x1, y1, x2, y2) in boxes:
        # Ensure coordinates are within the frame dimensions
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        
        # Crop the region of interest (ROI) from the frame
        cropped = frame[y1:y2, x1:x2]

        p_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        transform_list = [
            transforms.Resize(size=target_size,interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        test_transform=transforms.Compose(transform_list)
        p_img = Image.fromarray(p_img)
        t_p_img=test_transform(p_img)
        
        # Append the resized cropped image to the list
        cropped_images.append(t_p_img)
    
    stacked_crops = torch.stack(cropped_images, dim=0)
    
    return stacked_crops

def extract_feature(model,img):
    #img = img.to(device)
    #utput = model(img)
    #return output.detach().cpu()

    input_img = img.to(device)
    outputs = model(input_img)

    f1 = outputs.data.cpu()
    # flip
    img = img.index_select(3, torch.arange(img.size(3) - 1, -1, -1))
    input_img = img.to(device)
    outputs = model(input_img)
    f2 = outputs.data.cpu()

    ff = f1 + f2

    return ff.detach().cpu()


def custom_argmax(array):
    # Apply argmax along axis 1
    max_indices = np.argmax(array, axis=1)
    
    # Create a mask for rows where all values are zero
    zero_rows_mask = np.all(array == 0, axis=1)
    
    # Replace indices in rows where all values are zero with -1
    max_indices[zero_rows_mask] = -1
    
    return max_indices

def add_id_to_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    # Get properties of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    old_people_features_list=[]

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    calculation_duration = 2*fps  # seconds to run calculation
    skip_duration = 5*fps  # seconds to skip calculation
    frame_counter = 0
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if 1:#(frame_counter % (calculation_duration + skip_duration)) < calculation_duration:

            y_hat=det_model.predict(frame, conf = conf_level, classes = class_IDS, device = device, verbose = False)
            
            boxes = y_hat[0].boxes.xyxy.cpu().numpy()
            confidences = y_hat[0].boxes.conf.cpu().numpy()
            classes = y_hat[0].boxes.cls.cpu().numpy()

            if len(boxes)!=0 and len(confidences)!=0:
                print("-----------------")
                croped_p_images=crop_and_resize_persons(frame,boxes)

                features = extract_feature(feature_model, croped_p_images)

                B, T, D = features.shape  # B: Batch size, T: Sequence length (14), D: Feature dimension (768)
                fnorm = torch.norm(features, p=2, dim=2, keepdim=True) * np.sqrt(T)  # Shape: (B, 14, 1)
                features_norm = features / fnorm  # Shape: (B, 14, 768)
                # Use concatenated_vectors = features_norm.view(B, -1).detach().numpy() if you want features in (B, 10752) instead of (B, 768)
                concatenated_vectors = features_norm.mean(dim=1).detach().numpy()

                print("old_people_features_list:",len(old_people_features_list))

                people_id=[]
                cat_flag=False
                if len(old_people_features_list)!=0:
                    old_people_features = np.concatenate(old_people_features_list, axis=0)
                    similarity_score=cosine_similarity(concatenated_vectors,old_people_features)
                    similarity_score[similarity_score < threshold] = 0
                    people_id = custom_argmax(similarity_score).tolist()

                    conditions_tensor=np.array(people_id)
                    mask = conditions_tensor == -1
                    print("mask",mask.shape)
                    print("concatenated_vectors",concatenated_vectors.shape)
                    filtered_concatenated_vectors = concatenated_vectors[mask]

                    if filtered_concatenated_vectors.ndim == 1:
                            filtered_concatenated_vectors=filtered_concatenated_vectors[None]

                    cat_flag=True

                    print("old_people_features_list is not empty",people_id,len(filtered_concatenated_vectors)!=0)
                
                if cat_flag:
                    if len(filtered_concatenated_vectors)!=0:
                        old_people_features_list.append(filtered_concatenated_vectors)
                        print("filtered_concatenated_vectors shape",filtered_concatenated_vectors.shape)
                    else:
                        pass
                else:
                    old_people_features_list.append(concatenated_vectors)
                    print("concatenated_vectors shape",concatenated_vectors.shape)
                old_people_features = np.concatenate(old_people_features_list, axis=0)
                print("old_people_features:",old_people_features.shape)
                
                if len(people_id)!=0:
                    print(people_id)
                    for i,(p_id,find) in enumerate(zip(people_id,concatenated_vectors)):
                        if p_id==-1:
                            print("find for -1")
                            people_id[i]=np.where(old_people_features==find)[0][0]
                    print("people_id",people_id)
                else:
                    print("people_id len 0")
                    for find in concatenated_vectors:
                        people_id.append(np.where(old_people_features==find)[0][0])
                    print("people_id",people_id)

                frame=draw_bounding_boxes_with_ids(frame, boxes, people_id)

        # Write the modified frame to the output video
        out.write(frame)

        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_counter += 1
    
    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

input_path=r"Video\In\Pedestrians walk over a front car hood that stop in the pedestrian lane.mp4"
output_path=r"Video\Out\fliped"
add_id_to_video(input_path, os.path.join(output_path,os.path.basename(input_path)))