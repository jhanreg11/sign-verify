from torchvision.transforms import ToTensor
import os, torch, random, base64, numpy as np
from PIL import Image, ImageOps
from io import BytesIO
from server.model.model import SiameseConvNet
import uuid

IMAGE_SIZE = (220, 155)
path = os.getcwd() + '/server'
pil2tensor = ToTensor()


def load_model():
    """Loads and returns pretrained model"""
    device = torch.device('cpu')
    model = SiameseConvNet().eval()
    model.load_state_dict(torch.load(os.path.join(path, 'model/weights/model_epoch_20'), map_location=device))
    return model

def create_user(name, signatures):
    """creates a new directory for a user and seeds real signatures into <username>/real"""
    new_path = os.join(path, 'data', name)
    # creates new directory for user signatures
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, 'real'))
    os.mkdir(os.path.join(new_path, 'fake'))
    for sig in signatures:
        save_image(os.path.join(new_path, 'real', sig))

def prepare(input, baseline=True):
    """returns correct tensor represantion of input, which can be a string, 2d array or base64 string"""
    if type(input) == str and baseline:
        dir_path = os.path.join(path, 'data', input, 'real')
        print('\n\n', input, '\n\n\n', dir_path)
        return load_image(os.path.join(dir_path, random.choice(os.listdir(dir_path))))
    # prepares encoded string representation
    return pil2tensor(ImageOps.invert(str_to_pil(input)).convert('L').resize(IMAGE_SIZE, resample=Image.BILINEAR)) / 5.0

def load_images(path):
    """loads, normalizes, and inverts all images from a directory"""
    images = []
    for img in os.listdir(path):
        images.append(load_image(os.path.join(path, 'data', img)))
    return images

def load_image(path):
    return pil2tensor(Image.open(path)) / 255.0

def add_training_data(user, signature, classification):
    dir_path = os.path.join(path, 'data', user, 'real' if classification else 'fake')
    save_image(dir_path, signature)

def save_image(path, signature):
    img = ImageOps.invert(str_to_pil(signature).resize(IMAGE_SIZE, resample=Image.BILINEAR)).convert('L')
    img.save(os.path.join(path, 'real', '{}.png'.format(uuid.uuid1().hex)), 'PNG')

def str_to_pil(sig):
    lst = [0 for _ in range(sig.keys())]
    for key in sig.keys():
        lst[int(key)] = []
        for val in sig[key]:
            lst[int(key)].append(BytesIO(base64.b64decode(val)))
    return Image.fromarray(np.array(lst))