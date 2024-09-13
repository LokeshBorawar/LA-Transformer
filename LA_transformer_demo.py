import timm
from LATransformer.model import ClassBlock, LATransformer, LATransformerTest
import torch
import os
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

device="cpu"

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base= vit_base.to(device)

# Create La-Transformer
model = LATransformerTest(vit_base, lmbd=8).to(device)

# Load LA-Transformer
save_path = os.path.join('./model','net_best.pth')
model.load_state_dict(torch.load(save_path), strict=False)
model.eval()

def extract_feature(model,img):
    img = img.to(device)
    output = model(img)
    return output.detach().cpu()


def load_image(path):
    im=cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    transform_gallery_list = [
        transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    test_transform=transforms.Compose(transform_gallery_list)
    im = Image.fromarray(im)
    im = test_transform(im)
    return im

person_data=[]
for image_name in sorted(os.listdir("Persons_imgs\Imgs")):
    image_path="Persons_imgs/Imgs/"+image_name
    person_data.append(load_image(image_path))
person_data=torch.stack(person_data, dim=0)
print("inputs shape:",person_data.shape)

features = extract_feature(model, person_data)
print("features shape:",features.shape)

B, T, D = features.shape  # B: Batch size, T: Sequence length (14), D: Feature dimension (768)
fnorm = torch.norm(features, p=2, dim=2, keepdim=True) * np.sqrt(T)  # Shape: (B, 14, 1)
features_norm = features / fnorm  # Shape: (B, 14, 768)
# Use concatenated_vectors = features_norm.view(B, -1).detach().numpy() if you want features in (B, 10752) instead of (B, 768)
concatenated_vectors = features_norm.mean(dim=1).detach().numpy()

x=np.array(concatenated_vectors)
print("Processed features shape:",x.shape)
cm=cosine_similarity(x,x)

# Plot and save the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=sorted(os.listdir("Persons_imgs\Imgs")), yticklabels=sorted(os.listdir("Persons_imgs\Imgs")))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix
plt.savefig('Persons_imgs/confusion_matrix.png')  # Save as an image file
plt.show()