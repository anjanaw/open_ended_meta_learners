import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pm_tg as tg
import os
import math
import argparse
import datetime as dt

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-tp", "--test_person", type=str, default='03')
parser.add_argument("-w", "--class_num", type=int, default=7)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-e", "--episode", type=int, default=100)
parser.add_argument("-t", "--test_episode", type=int, default=100)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-R", "--remove_class", type=int, default=0)

args = parser.parse_args()

# Hyper Parameters
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
TEST_PERSON = args.test_person
REMOVE_CLASS = [str(f) for f in np.random.choice(range(CLASS_NUM), args.remove_class, False)]

if torch.cuda.is_available():
    dev = "cuda:2"
else:
    dev = "cpu"
device = torch.device(dev)


def write_results(text):
    file_path = 'pm.csv'
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(text + '\n')
    else:
        f = open(file_path, 'w')
        f.write(text + '\n')
    f.close()


class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, ssp_y):
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(ssp_y.float()).squeeze()
        return preds


class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            cosine_similarity = torch.cosine_similarity(support_image, input_image, dim=1, eps=eps)
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1280, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x.float())
        return out


class RelationNetwork(nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = DistanceNetwork()
        self.layer2 = AttentionalClassify()

    def forward(self, ssp, inp, ssp_y):
        sims = self.layer1(ssp, inp)
        out = self.layer2(sims, ssp_y)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def byclass(predl, truel, norm):
    rewards = np.zeros(len(set(truel)))
    for py, ty in zip(predl, truel):
        if py == ty:
            rewards[ty] += 1
    rewards = rewards/norm
    return rewards, sum(rewards)/len(rewards)


def main():
    write_results(str(args)+str(REMOVE_CLASS))
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.act_folders(TEST_PERSON)

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = Encoder()
    relation_network = RelationNetwork()

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.to(device)
    relation_network.to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    # relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    test_accuracies = []
    byclass_accuracies = []
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        # relation_network_scheduler.step(episode)

        task = tg.ACTTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, split="train", remove=REMOVE_CLASS)
        sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)
        batch_dataloader = tg.get_data_loader(task, num_per_class=task.per_class_num, split="test", shuffle=True)

        # sample datas
        samples, sample_labels = sample_dataloader.__iter__().next()
        # print('samples.shape', samples.shape)
        batches, batch_labels = batch_dataloader.__iter__().next()
        # print('batches.shape', batches.shape)

        # calculate features
        sample_features = feature_encoder(Variable(samples).to(device))
        # print('sample_features.shape', sample_features.shape)
        sample_features = sample_features.view(task.num_classes, SAMPLE_NUM_PER_CLASS, 1200)
        # print('sample_features.shape', sample_features.shape)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        # print('sample_features.shape', sample_features.shape)
        batch_features = feature_encoder(Variable(batches).to(device))
        # print('batch_features.shape', batch_features.shape)

        sample_features_ext = sample_features.unsqueeze(1).repeat(1, batch_features.shape[0], 1)
        # print('sample_features_ext.shape', sample_features_ext.shape)
        sample_labels = sample_labels.view(task.num_classes, SAMPLE_NUM_PER_CLASS)
        # print('sample_labels.shape', sample_labels.shape)
        sample_labels = torch.unique(sample_labels)
        # print('sample_labels.shape', sample_labels.shape)
        sample_labels_ext = sample_labels.unsqueeze(0).repeat(batch_features.shape[0], 1)
        # print('sample_labels_ext.shape', sample_labels_ext.shape)
        sample_labels_ext = torch.nn.functional.one_hot(sample_labels_ext).to(device)
        # print('sample_labels_ext.shape', sample_labels_ext.shape)

        relations = relation_network(sample_features_ext, batch_features, sample_labels_ext)
        # print('relations.shape', relations.shape)

        cce = nn.CrossEntropyLoss().to(device)
        # print('batch_labels.shape:', batch_labels.shape)
        loss = cce(relations, batch_labels.to(device))


        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        # relation_network_optim.step()

        print("episode:", episode + 1, "loss", loss.item())
        # write_results('episode:'+str(episode + 1)+'loss'+str(loss.item()))

        if (episode + 1) % 5 == 0:

            # test
            print("Testing...")
            total_rewards = 0
            predicted_total = []
            true_total = []
            class_num = 0
            for i in range(TEST_EPISODE):
                task = tg.ACTTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, split="test", remove=REMOVE_CLASS)
                sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                       shuffle=False)
                test_dataloader = tg.get_data_loader(task, num_per_class=task.per_class_num, split="test",
                                                     shuffle=False)

                sample_images, sample_labels = sample_dataloader.__iter__().next()
                # print('sample_images.shape', sample_images.shape)
                test_images, test_labels = test_dataloader.__iter__().next()
                # print('test_images.shape', test_images.shape)
				
                # calculate features
                sample_features = feature_encoder(Variable(sample_images).to(device))
                # print('sample_features.shape', sample_features.shape)
                sample_features = sample_features.view(task.num_classes, SAMPLE_NUM_PER_CLASS, 1200)
                # print('sample_features.shape', sample_features.shape)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                # print('sample_features.shape', sample_features.shape)
                test_features = feature_encoder(Variable(test_images).to(device))
                # print('test_features.shape', test_features.shape)

                sample_features_ext = sample_features.unsqueeze(1).repeat(1, test_features.shape[0], 1)
                # print('sample_features_ext.shape', sample_features_ext.shape)
                sample_labels = sample_labels.view(task.num_classes, SAMPLE_NUM_PER_CLASS)
                # print('sample_labels.shape', sample_labels.shape)
                sample_labels = torch.unique(sample_labels)
                # print('sample_labels.shape', sample_labels.shape)
                sample_labels_ext = sample_labels.unsqueeze(0).repeat(test_features.shape[0], 1)
                # print('sample_labels_ext.shape', sample_labels_ext.shape)
                sample_labels_ext = torch.nn.functional.one_hot(sample_labels_ext).to(device)
                # print('sample_labels_ext.shape', sample_labels_ext.shape)

                relations = relation_network(sample_features_ext, test_features, sample_labels_ext)
                # print('relations.shape', relations.shape)

                _, predict_labels = torch.max(relations.data, 1)
                predicted_total.extend(predict_labels.tolist())
                true_total.extend(test_labels.tolist())
                class_num += task.per_class_num
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in
                           range(CLASS_NUM * task.per_class_num)]

                total_rewards += np.sum(rewards)

            byclass_accuracy, total_accuracy = byclass(predicted_total, true_total, class_num)
            print("byclass accuracy:", byclass_accuracy)
            print("total accuracy:", total_accuracy)
            # test_accuracy = total_rewards / 1.0 / CLASS_NUM / task.per_class_num / TEST_EPISODE

            # print("test accuracy:", test_accuracy)
            test_accuracies.append(str(total_accuracy))
            byclass_accuracies.extend(byclass_accuracy)

    write_results(','.join(test_accuracies))
    write_results(','.join([str(f) for f in byclass_accuracies]))


if __name__ == '__main__':
    main()
