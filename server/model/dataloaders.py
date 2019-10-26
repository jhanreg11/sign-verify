from server.model.preprocessor import load_images
import os
from itertools import combinations
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

path = os.getcwd() + '/server/data'

def gen_pairs(path):
    dirs = os.listdir(path)
    pairs = []
    for dir in dirs:
        real_images = []
        fake_images = []

        try:
            real_images = load_images(os.path.join(path, dir, 'real'))
            fake_images = load_images(os.path.join(path, dir, 'fake'))
        except:
            print('error retrieving images')
        positives = combinations(real_images, 2)
        pairs += [[p[0], p[1], 1] for p in positives]
        for real in real_images:
            for fake in fake_images:
                pairs.append([real, fake, 0])

        return pairs

train, test = train_test_split(gen_pairs(path), test_size=.35)

class TrainDataset(Dataset):
    def __init__(self):
        self.pairs = train

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

class TestDataset(Dataset):
    def __init__(self):
        self.pairs = test

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)