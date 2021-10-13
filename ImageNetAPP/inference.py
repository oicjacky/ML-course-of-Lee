import io
import json
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

import pdb

MODEL = models.densenet121(pretrained=True) # pretrained with [imagenet](https://pytorch.org/vision/0.8/datasets.html#imagenet)
MODEL.eval() # since we are using our model only for inference, switch to `eval` mode
IMAGENET_CLASS_INDEX = json.load(open('./imagenet_class_index.json')) # [source of dataset](https://image-net.org/challenges/LSVRC/2012/)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # channel_1~3 = RGB and channel_4 = transparency
    return my_transforms(image).unsqueeze(0)

def tensor_to_image(tensor, use_matplotlib = False):
    import numpy as np
    import matplotlib.pyplot as plt
    if tensor.ndim > 3:
        tensor = np.squeeze(tensor, axis=0)
    print(tensor, tensor.shape)
    img = transforms.ToPILImage()(tensor)
    if not use_matplotlib:
        img.show()
    else:
        plt.imshow(img)
        plt.title(IMAGE_FILE_PATH)
        plt.show()

def get_prediction(image_bytes, model = MODEL, imagenet_class_index = IMAGENET_CLASS_INDEX):
    tensor = transform_image(image_bytes)
    outputs= model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


if __name__ == "__main__":

    # r"C:\Users\user\Pyvirtualenv\course_CNN\images\Machine-Learning-HW4-RNN_1.PNG"
    IMAGE_FILE_PATH = r'C:\Users\user\Pyvirtualenv\course_CNN\images\hi-res-9021ebe2b61de5b19dbe52f63b3e651c_crop_north.jpg'

    with open(IMAGE_FILE_PATH, 'rb') as f:
        image_bytes = f.read()
        
        tensor = transform_image(image_bytes)
        tensor_to_image(tensor)

        print(get_prediction(image_bytes))