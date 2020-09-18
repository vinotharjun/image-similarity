import torch
from config import *
import torch.nn as nn
from torchvision import transforms
from numpy import dot
from numpy.linalg import norm
from torchvision import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle
# def load_weights(model=model):
#     loaded_data = torch.load("./saved_models//saved_model_v3.pth",map_location={"cuda:0":"cpu"})
#     model.load_state_dict(loaded_data)
#     return model
# embedding_net = load_weights(model)
# embedding_net.eval()
# def transform_image(image):
#     my_transforms = transforms.Compose([transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,),(0.5,))])
#     image_tensor = my_transforms(image).unsqueeze(0)
#     return image_tensor

# def prediction(image):
#     tensor = transform_image(image=image)
#     output = embedding_net.get_embedding(tensor).squeeze()
# #     print(output.shape)
#     return output
# def make_storage(database_path):
#     storage = {}
#     for i in os.listdir(database_path):
#         path= database_path+"/"+i
#         pil = Image.open(path)
#         feature = prediction(pil).squeeze().cpu().detach().numpy()
#         storage[path] = feature
#     return storage

def load_storage():
    with open('data.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b
def cosine_similarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def make_search(storage,query):
    distance = {}
    for i,feature in storage.items():
        # print(i)
        distance[i] = cosine_similarity(feature,query)
    return distance

def closest_pairs(distance):
    va =[]
    for i,value in distance.items():
        if value>0.6:
            va.append({"image":i,"score":value})
    newlist=sorted(va, key = lambda k:k['score'], reverse=True)
    return newlist
        
def main_search(query):
    storage = load_storage()
    distance = make_search(storage,query)
    closest = closest_pairs(distance)
    return closest
    
