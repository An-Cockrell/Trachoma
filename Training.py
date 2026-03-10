import os
import time
import csv
import json
import numpy as np
import wandb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
import torchmetrics

from DataLoader_lightning import TrachomaDataModule, ToTensor, CustomCrop, FollicleEnhance

# torch.multiprocessing.set_start_method('spawn')


class Training:
    def __init__(self, dataloaders, net, save_info, lr=5e-3, epochs=10):
        # generate folder to save run
        os.mkdir(save_info)

        # start new wandb run
        wandb.init(project='Trachoma', entity='dsocia22')
        # config = wandb.config
        # config.learning_rate = 0.01
        # config.batch_size = dataloaders['train'].batch_size
        # config.num_workers = dataloaders['train'].num_workers
        wandb.run.name = save_info
        # wandb.run.save()

        # self.train_data, self.val_data, self.test_data = dataloaders['train'], dataloaders['val'], dataloaders['test']
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.save_info = save_info

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

            if torch.cuda.device_count() > 1:
                print('Using {} GPUs'.format(torch.cuda.device_count()))
                net = nn.DataParallel(net)

        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        self.net = net
        self.net.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)


        self.stats = {'train_loss': [], 'train_acc': [], 'train_prec': [], 'train_f1': [], 'train_recall': [],
                      'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_f1': [], 'val_recall': []}

        self.best_loss = float('inf')

        print('Initialized')

    def train(self, epoch):
        start = time.strftime('%H:%M:%S')
        print('Epoch {}/{} | Start Time : {}'.format(epoch+1, self.epochs, start))
        print('_' * 10)

        stats = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                self.net.train()
                # dataloader = self.dataloaders[phase]
                dataloader = self.dataloaders.train_dataloader()
            else:
                self.net.eval()
                dataloader = self.dataloaders.val_dataloader()

            running_loss = 0.0
            running_acc = 0.0
            running_pre = 0.0
            running_recall = 0.0
            running_f1 = 0.0

            step = 0

            wandb.watch(self.net)
            for i, samples in enumerate(dataloader):
                imgs = samples['image'].float().to(self.device)
                target = samples['label'].to(self.device)

                step += 1

                if phase == 'train':
                    # zero the gradients
                    self.optimizer.zero_grad()
                    outputs = self.net(imgs).squeeze()
                    loss = self.criterion(outputs, target.float())

                    loss.backward()
                    self.optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = self.net(imgs).squeeze()
                        loss = self.criterion(outputs, target.float())

                outputs = F.log_softmax(outputs, dim=1)
                _, outputs = torch.max(outputs, dim=1)

                outputs = outputs.cpu().detach().numpy()
                target = target.cpu().detach().numpy()

                # print('Pred  : ', outputs)
                # print('Target: ', target)

                acc = metrics.accuracy_score(target, outputs)
                pre = metrics.precision_score(target, outputs, zero_division=0)
                recall = metrics.recall_score(target, outputs, zero_division=0)
                f1 = metrics.f1_score(target, outputs, zero_division=0)

                # torch.cuda.empty_cache()

                running_acc += acc * len(target)
                running_loss += loss.item() * len(target)
                running_pre += pre * len(target)
                running_recall += recall * len(target)
                running_f1 += f1 * len(target)

                if step % 10 == 0:
                    print('Current step: {} Loss: {} Acc: {} Precision: {} Recall: {} F1: {}'.format(step, loss, acc, pre, recall, f1))

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            epoch_pre = running_pre / len(dataloader.dataset)
            epoch_recall = running_recall / len(dataloader.dataset)
            epoch_f1 = running_f1 / len(dataloader.dataset)

            print('Phase: {} Loss: {} Acc: {} Precision: {} Recall: {} F1: {}'.format(phase, epoch_loss, epoch_acc, epoch_pre, epoch_recall, epoch_f1))


            if phase == 'train':
                stats['train_loss'] = epoch_loss
                stats['train_acc'] = epoch_acc
                stats['train_prec'] = epoch_pre
                stats['train_f1'] = epoch_f1
                stats['train_recall'] = epoch_recall

            else:
                stats['val_loss'] = epoch_loss
                stats['val_acc'] = epoch_acc
                stats['val_prec'] = epoch_pre
                stats['val_f1'] = epoch_f1
                stats['val_recall'] = epoch_recall

        wandb.log(stats)

        for key, value in stats.items():
            self.stats[key].append(value)

        with open('./{}/training_model_stats.csv'.format(self.save_info), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.stats.keys())
            writer.writerows(zip(*self.stats.values()))

    def start(self):
        last_improvment = 0
        for epoch in range(self.epochs):
            self.train(epoch)

            self.scheduler.step(self.stats['val_loss'][-1])

            if self.stats['val_loss'][-1] < self.best_loss:
                last_improvment = 0
                state = {
                    'epoch': epoch,
                    'best_loss': self.best_loss,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                print('*' * 10 + ' New Optimal Found, Saving State ' + '*' * 10)
                state['best_loss'] = self.best_loss = self.stats['val_loss'][-1]
                torch.save(state, './{}/model_best.pth'.format(self.save_info))
            else:
                last_improvment += 1

            state = {
                'epoch': epoch,
                'loss': self.stats['val_loss'][-1],
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            torch.save(state, './{}/model_epoch{}.pth'.format(self.save_info, epoch))

            if last_improvment >= 5:
                print('No improvement in 5 epochs cancelling training')
                break

        print('***** Testing *****')
        self.net.eval()
        dataloader = self.dataloaders.test_dataloader()

        running_loss = 0.0
        running_acc = 0.0
        running_pre = 0.0
        running_recall = 0.0
        running_f1 = 0.0

        step = 0

        confusion_matrix = torch.zeros([2, 2]).to(self.device)

        for i, samples in enumerate(dataloader):
            imgs = samples['image'].float().to(self.device)
            target = samples['label'].to(self.device)

            step += 1

            with torch.no_grad():
                outputs = self.net(imgs).squeeze()
                loss = self.criterion(outputs, target.float())

                outputs = F.softmax(outputs, dim=1)
                _, outputs = torch.max(outputs, dim=1)

                inds = torch.stack((target, outputs))
                u, c = torch.unique(inds, dim=1, return_counts=True)
                confusion_matrix[u[0, :], u[1, :]] += c

            outputs = outputs.cpu().detach().numpy()
            target = target.cpu().detach().numpy()

            # print('Pred  : ', outputs)
            # print('Target: ', target)

            acc = metrics.accuracy_score(target, outputs)
            pre = metrics.precision_score(target, outputs, zero_division=0)
            recall = metrics.recall_score(target, outputs, zero_division=0)
            f1 = metrics.f1_score(target, outputs, zero_division=0)

            # torch.cuda.empty_cache()

            running_acc += acc * dataloader.batch_size
            running_loss += loss.item() * dataloader.batch_size
            running_pre += pre * dataloader.batch_size
            running_recall += recall * dataloader.batch_size
            running_f1 += f1 * dataloader.batch_size


        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_acc / len(dataloader.dataset)
        epoch_pre = running_pre / len(dataloader.dataset)
        epoch_recall = running_recall / len(dataloader.dataset)
        epoch_f1 = running_f1 / len(dataloader.dataset)

        print('Phase: {} Loss: {} Acc: {} Precision: {} Recall: {} F1: {}'.format('test', epoch_loss, epoch_acc, epoch_pre,
                                                                                epoch_recall, epoch_f1))
        # save confusion matrix
        with open('./{}/test_confusion_matix.npy'.format(self.save_info), 'wb') as f:
            confusion_matrix = confusion_matrix.cpu().detach().numpy()
            np.save(f, confusion_matrix)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    img_keys = 'TrachomaData/trachomagroundtruthkey.csv'

    trans_0 = transforms.Compose(
        [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.Resize(256),
         transforms.CenterCrop(224)])
    trans_1 = transforms.Compose(
        [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
         transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
         transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
                            batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    dm.setup()

    res101 = models.resnet101(pretrained=True)
    # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    print(res101)  # 1000

    # Newly created modules have require_grad=True by default
    num_features = res101.fc.in_features
    # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    print(res101)

    # classifier12 = TrachomaClassifier(res101)

    #
    run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_allLayers_pretrained_norm_resnet2'
    trainer = Training(dm, res101, save_info=run_info)
    trainer.start()
    # run_experiment(run_info, dm, classifier12)
