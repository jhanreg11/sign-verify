from torchvision.transforms import ToTensor
import os, torch, random, base64
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
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, 'real'))
    os.mkdir(os.path.join(new_path, 'fake'))
    for sig in signatures:
        save_image(os.path.join(os.path.join(new_path, 'real'), sig))

def prepare(input, baseline=True):
    """returns correct tensor represantion of input, which can be a string, 2d array or base64 string"""
    if type(input) == str and baseline:
        dir_path = os.path.join(path, 'data', input, 'real')
        print('\n\n', input, '\n\n\n', dir_path)
        return load_image(os.path.join(dir_path, random.choice(os.listdir(dir_path))))
    return pil2tensor(ImageOps.invert(Image.open(BytesIO(base64.b64decode(input)))).convert('L').resize(IMAGE_SIZE, resample=Image.BILINEAR))

def load_images(path):
    """loads, normalizes, and inverts all images from a directory"""
    images = []
    for img in os.listdir(path):
        images.append(load_image(os.path.join(path, 'data', img)))
    return images

def load_image(path):
    img = ImageOps.invert(Image.open(path)).convert('L')
    resized = img.resize(IMAGE_SIZE, resample=Image.BILINEAR)
    return pil2tensor(resized) / 255.0

def add_training_data(user, signature, classification):
    dir_path = os.path.join(path, 'data', user, 'real' if classification else 'fake')
    save_image(dir_path, signature)

def save_image(path, signature):
    img = Image.open(BytesIO(base64.b64decode(signature))).resize(IMAGE_SIZE, resample=Image.BILINEAR)
    img.save(os.path.join(path, 'real', f'{uuid.uuid1().hex}.png'), 'PNG')
