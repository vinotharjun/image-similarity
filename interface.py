import io 
import torchvision.transforms as transforms
from PIL import Image
from model import *
from config import *
import cv2
from PIL import Image
import numpy as np
import cv2

embedder = embedding_net()

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])
    image_tensor = my_transforms(image).unsqueeze(0)
    return image_tensor
def prediction(image):
    tensor = transform_image(image=image)
    output = embedder.get_embedding(tensor)
    return output

# img = cv2.imread("/media/vinodarjun/Storage/deeplearning Projects/computer vision/oneshot/imagetest.jpeg")
# img =Image.fromarray(img)

# print(prediction(img).shape)
# print(prediction(img))
