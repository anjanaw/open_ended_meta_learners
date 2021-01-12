import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import csv
import numpy as np
from torch.utils.data.sampler import Sampler


def act_folders(test_person):
    data_folder = '/home/mex/data/act_maml/'

    metatrain_character_folders = {}
    metaval_character_folders = {}
    persons = [f for f in os.listdir(data_folder) if f != '.DS_Store']
    test_persons = np.random.choice(persons, int(len(persons)/3), False)
    for person in [f for f in os.listdir(data_folder) if f != '.DS_Store']:
        person_folder = os.path.join(data_folder, person)
        if person in test_persons:
            metaval_character_folders[person] = {}
        else:
            metatrain_character_folders[person] = {}
        for activity in [p for p in os.listdir(person_folder) if p != '.DS_Store']:
            activity_folder = os.path.join(person_folder, activity)
            if person in test_persons:
                metaval_character_folders[person][activity] = []
            else:
                metatrain_character_folders[person][activity] = []
            for item in [a for a in os.listdir(activity_folder) if a != '.DS_Store']:
                if person in test_persons:
                    metaval_character_folders[person][activity].append(os.path.join(activity_folder, item))
                else:
                    metatrain_character_folders[person][activity].append(os.path.join(activity_folder, item))
    return metatrain_character_folders, metaval_character_folders


class ACTTask(object):
    def __init__(self, character_folders, num_classes, train_num, split="train", remove=[]):
        data_folder = '/home/mex/data/act_maml/'
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        # self.test_num = test_num

        random_person = random.sample(self.character_folders.keys(), 1)[0]
        random_person_path = os.path.join(data_folder, random_person)
        class_folders = random.sample(self.character_folders[random_person].keys(), self.num_classes)
        if split == "train":
            class_folders = [f for f in class_folders if f not in remove]
        class_folders = [os.path.join(random_person_path, act) for act in class_folders]
        self.num_classes = len(class_folders)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        self.per_class_num = 10000
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            if self.per_class_num > len(temp) - train_num:
                self.per_class_num = len(temp) - train_num

        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+self.per_class_num]

        self.test_num = self.per_class_num

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return '/'+os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class ACT(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(ACT, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        _data = csv.reader(open(image_root, "r"), delimiter=",")
        for row in _data:
            image = [float(f) for f in row]
            image = np.array(image)
            image = np.reshape(image, (1, 5, 180))
            if self.transform is not None:
                image = self.transform(image)
            label = self.labels[idx]
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train', shuffle=True):

    dataset = ACT(task, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader