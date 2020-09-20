import torch
from config import *
import torch.nn as nn
from torchvision import transforms

from torchvision import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_model():
    #loading the model
    model_50 = models.resnet50(pretrained=True)
    #defining the architecute
    for param in model_50.parameters():
        param.requires_grad = False
        n_inputs = model_50.fc.in_features
        last_layer = nn.Linear(n_inputs, 128)
        model_50.fc = last_layer
    model_50.to(device)
    return model_50
def load_other_feature_extractor(model_type):
    if model_type == "resnet18":
        fextractor = models.resnet18(pretrained=True)
    elif model_type =="vgg16":
        fextractor = models.vgg16(pretrained=True)
    elif model_type == "resnet34":
        fextractor = models.resnet34(pretrained=True)
    features_model = nn.Sequential(*list(fextractor.children())[:-1])
    return features_model



class EmbeddingNet(nn.Module):
    def __init__(self,pretrained_net):
        super(EmbeddingNet, self).__init__()
        self.resnet = pretrained_net

    def forward(self, x):
        output = self.resnet(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)
class TripletNet(nn.Module):
    '''
    input
    embedding net : ConvNet which takes torch.tensor input
     run parallel convnet for each batch
    '''
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        return self.embedding_net(x)
    def get_embedding(self, x):
        return self.embedding_net(x)        




def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])
    image_tensor = my_transforms(image).unsqueeze(0)
    return image_tensor
res = load_model()
embedding_net = EmbeddingNet(res)
model = TripletNet(embedding_net)


def load_weights(model=model):
    loaded_data = torch.load(MODEL_WEIGHTS_PATH,map_location={"cuda:0":"cpu"})
    model.load_state_dict(loaded_data)
    return model
embedding_net = load_weights(model)
embedding_net.eval()

def prediction(image,model_type="tripplet"):
    if model_type!="tripplet":
        fextractor = load_other_feature_extractor(model_type)
        tensor = transform_image(image=image)
        output = fextractor(tensor.to(device))
        # print(output.shape)
        if "vgg" in model_type:
            output = torch.nn.AdaptiveAvgPool2d((1))(output)
        output = output.squeeze()
        # print(output.shape)
        return output
    else:
        tensor = transform_image(image=image)
        output = embedding_net.get_embedding(tensor).squeeze()
        # print(output.shape)
        return output
 




# loaded_data = torch.load(MODEL_WEIGHTS_PATH,map_location={"cuda:0":"cpu"})
# model.load_state_dict(loaded_data)
# for k,_ in loaded_data.items():
#     print(k)