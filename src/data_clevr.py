import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import random
random.seed(10)


def load_images_and_labels(dataset='clevr-hans3', split='train'):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    labels = []
    folder = 'data/clevr/' + dataset + '/' + split + '/'
    true_folder = folder + 'true/'
    false_folder = folder + 'false/'

    filenames = sorted(os.listdir(true_folder))
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(true_folder, filename))
            labels.append(1)

    filenames = sorted(os.listdir(false_folder))
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(false_folder, filename))
            labels.append(0)
    return image_paths, labels


def load_images_and_labels_positive(dataset='clevr-hans0', split='train'):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    labels = []
    folder = 'data/clevr/' + dataset + '/' + split + '/'
    true_folder = folder + 'true/'
    false_folder = folder + 'false/'

    filenames = sorted(os.listdir(true_folder))[:500]
    #n = 500  # int(len(filenames)/10)
    #filenames = random.sample(filenames, n)
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(true_folder, filename))
            labels.append(1)
    return image_paths, labels

def __load_images_and_labels(dataset='clevr-hans0', split='train', base=None):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    labels = []
    if base == None:
        base_folder = 'data/clevr/' + dataset + '/' + split + '/'
    else:
        base_folder = base + '/data/clevr/' + dataset + '/' + split + '/'
    if dataset == 'clevr-hans3':
        for i, cl in enumerate(['class0', 'class1', 'class2']):
            folder = base_folder + cl + '/'
            filenames = sorted(os.listdir(folder))
            for filename in filenames:
                if filename != '.DS_Store':
                    image_paths.append(os.path.join(folder, filename))
                    labels.append(i)
    elif dataset == 'clevr-hans7':
        for i, cl in enumerate(['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6']):
            folder = base_folder + cl + '/'
            filenames = sorted(os.listdir(folder))
            for filename in filenames:
                if filename != '.DS_Store':
                    image_paths.append(os.path.join(folder, filename))
                    labels.append(i)
    return image_paths, labels


class CLEVRHans(torch.utils.data.Dataset):
    """CLEVRHans dataset. 
    The implementations is mainly from https://github.com/ml-research/NeSyConceptLearner/blob/main/src/pretrain-slot-attention/data.py.
    """

    def __init__(self, dataset, split, img_size=128, base=None):
        super().__init__()
        self.img_size = img_size
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size))]
        )
        self.image_paths, self.labels = load_images_and_labels(
            dataset=dataset, split=split)

    def __getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return image, label


    def __old__getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        if self.dataset == 'clevr-hans3':
            labels = torch.zeros((3, ), dtype=torch.float32)
        elif self.dataset == 'clevr-hans7':
            labels = torch.zeros((7, ), dtype=torch.float32)
        labels[self.labels[item]] = 1.0
        return image, labels

    def __len__(self):
        return len(self.labels)

class CLEVRHans_POSITIVE(torch.utils.data.Dataset):
    """CLEVRHans dataset.
    The implementations is mainly from https://github.com/ml-research/NeSyConceptLearner/blob/main/src/pretrain-slot-attention/data.py.
    """

    def __init__(self, dataset, split, img_size=128, base=None):
        super().__init__()
        self.img_size = img_size
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size))]
        )
        self.image_paths, self.labels = load_images_and_labels_positive(
            dataset=dataset, split=split)

    def __getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return image, label


    def __len__(self):
        return len(self.labels)

"""
class CLEVRConcept(torch.utils.data.Dataset):
    #The Concept-learning dataset for CLEVR-Hans.
    

    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
        self.data, self.labels = self.load_csv()
        print('concept data: ', self.data.shape, 'labels: ', len(self.labels))

    def load_csv(self):
        data = []
        labels = []
        pos_csv_data = pd.read_csv(
            'data/clevr/concept_data/' + self.split + '/' + self.dataset + '_pos' + '.csv', delimiter=' ')
        pos_data = pos_csv_data.values
        #pos_labels = np.ones((len(pos_data, )))
        pos_labels = np.zeros((len(pos_data, )))
        neg_csv_data = pd.read_csv(
            'data/clevr/concept_data/' + self.split + '/' + self.dataset + '_neg' + '.csv', delimiter=' ')
        neg_data = neg_csv_data.values
        #neg_labels = np.zeros((len(neg_data, )))
        neg_labels = np.ones((len(neg_data, )))
        data = torch.tensor(np.concatenate(
            [pos_data, neg_data], axis=0), dtype=torch.float32)
        labels = torch.tensor(np.concatenate(
            [pos_labels, neg_labels], axis=0), dtype=torch.float32)
        return data, labels

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.data)
"""