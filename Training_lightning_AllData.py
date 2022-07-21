import gc
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

from DataLoader_lightning import TrachomaDataModule, ToTensor, CustomCrop, FollicleEnhance


class TrachomaClassifier(pl.LightningModule):
    def __init__(self, model, optimizer_metric='val_loss', weight=None, threshold=0.5):
        super().__init__()

        self.model = model
        self.optimizer_metric = optimizer_metric
        # self.loss = nn.CrossEntropyLoss(weight=weight)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)

        # metrics
        self.train_acc = torchmetrics.Accuracy(threshold=threshold)
        self.val_acc = torchmetrics.Accuracy(threshold=threshold)
        self.test_acc = torchmetrics.Accuracy(threshold=threshold)

        self.train_precision = torchmetrics.Precision(multiclass=False, num_classes=1, threshold=threshold)
        self.val_precision = torchmetrics.Precision(multiclass=False, num_classes=1, threshold=threshold)
        self.test_precision = torchmetrics.Precision(multiclass=False, num_classes=1, threshold=threshold)

        self.train_recall = torchmetrics.Recall(multiclass=False, num_classes=1, threshold=threshold)
        self.val_recall = torchmetrics.Recall(multiclass=False, num_classes=1, threshold=threshold)
        self.test_recall = torchmetrics.Recall(multiclass=False, num_classes=1, threshold=threshold)

        self.train_f1 = torchmetrics.F1(multiclass=False, num_classes=1, threshold=threshold)
        self.val_f1 = torchmetrics.F1(multiclass=False, num_classes=1, threshold=threshold)
        self.test_f1 = torchmetrics.F1(multiclass=False, num_classes=1, threshold=threshold)

    def forward(self, x):
        classification = self.model(x)
        return classification

    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label']

        outputs = self.model(images).squeeze(1)
        loss = self.loss(outputs, targets.float())

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def training_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['loss']
        outputs = batch_parts['outputs']
        targets = batch_parts['targets']

        # log metrics
        self.train_acc(outputs, targets)
        self.train_precision(outputs, targets)
        self.train_recall(outputs, targets)
        self.train_f1(outputs, targets)

        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_pre', self.train_precision, on_step=True, on_epoch=True)
        self.log('train_rec', self.train_recall, on_step=True, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)

        loss = torch.sum(losses) / torch.numel(losses)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        # do something with both outputs

        #print(outputs, targets)
        return loss

    # def training_epoch_end(self, outs):
    #     # log epoch metrics
    #     self.log('train_acc_epoch', self.accuracy.compute())
    #     self.log('train_pre_epoch', self.precision.compute())
    #     self.log('train_rec_epoch', self.recall.compute())
    #     self.log('train_f1_epoch', self.f1.compute())

    def validation_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label']

        outputs = self.model(images).squeeze(1)
        loss = self.loss(outputs, targets.float())

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def validation_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['loss']
        outputs = batch_parts['outputs']
        targets = batch_parts['targets']

        # log metrics
        self.val_acc(outputs, targets)
        self.val_precision(outputs, targets)
        self.val_recall(outputs, targets)
        self.val_f1(outputs, targets)

        self.log('val_acc', self.val_acc)#, on_step=True, on_epoch=True)
        self.log('val_pre', self.val_precision)#, on_step=True, on_epoch=True)
        self.log('val_rec', self.val_recall)#, on_step=True, on_epoch=True)
        self.log('val_f1', self.val_f1) #, on_step=True, on_epoch=True)

        # do something with both outputs
        loss = torch.sum(losses) / torch.numel(losses)
        self.log('val_loss', loss)

        #print(outputs, targets)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label']

        outputs = self.model(images).squeeze(1)
        loss = self.loss(outputs, targets.float())

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def test_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['loss']
        outputs = batch_parts['outputs']
        targets = batch_parts['targets']

        # log metrics
        self.test_acc(outputs, targets)
        self.test_precision(outputs, targets)
        self.test_recall(outputs, targets)
        self.test_f1(outputs, targets)

        self.log('test_acc', self.test_acc) #, on_step=True, on_epoch=True)
        self.log('test_pre', self.test_precision) #, on_step=True, on_epoch=True)
        self.log('test_rec', self.test_recall) #, on_step=True, on_epoch=True)
        self.log('test_f1', self.test_f1)#, on_step=True, on_epoch=True)

        # do something with both outputs
        loss = torch.sum(losses) / torch.numel(losses)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        if self.optimizer_metric == 'val_loss':
            mode = 'min'
        else:
            mode = 'max'

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode=mode, patience=3, verbose=True),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': self.optimizer_metric,
            },
        }


def run_experiment(run_info, dataloader, model, project='Trachoma', accum_batches=1, swa=False):
    print('Running Experiment: ', run_info)

    # train
    wandb_logger = WandbLogger(project=project, name=run_info)
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath='Checkpoints/{}'.format(run_info), save_last=True, save_top_k=1, mode='min', monitor='val_loss')
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=2, log_every_n_steps=20, logger=wandb_logger, max_epochs=50,
                          default_root_dir='Checkpoints', accelerator='ddp', callbacks=[early_stop_callback, checkpoint_callback], accumulate_grad_batches=accum_batches, stochastic_weight_avg=swa)
    else:
        trainer = pl.Trainer(num_processes=1, log_every_n_steps=20, logger=wandb_logger, max_epochs=50, default_root_dir='Checkpoints', callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, dataloader)

    # test
    trainer.test()

    wandb.finish()

    del wandb_logger
    del trainer
    del model
    del dataloader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':

    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # # read training data mean and std for normalization
    # with open('Trachoma_dataset_stats.json', 'r') as f:
    #     data_stats = json.load(f)

    # transfer learning on VGG16
    # Load the pretrained model from pytorch
    #vgg16 = models.vgg16_bn()
    # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    #print(vgg16.classifier[6].out_features)  # 1000

    # Freeze training for all layers
    #for param in vgg16.features.parameters():
        # print(param)
    #    param.require_grad = False

    # Newly created modules have require_grad=True by default
    #num_features = vgg16.classifier[6].in_features
    #features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    #features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    #vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    #print(vgg16)

    #classifier = TrachomaClassifier(vgg16)


    # ########## Experiment 1 #########
    #trans = transforms.Compose([ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #          transforms.CenterCrop(224)])
    #dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                         num_workers=4)
    
    #run_info = 'Pytorch_lightning_baseline'
    
    #run_experiment(run_info, dm, classifier)



    ########## Experiment 2 #########
    #trans = transforms.Compose([ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #         transforms.CenterCrop(224)])
    #dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                        num_workers=4, oversample=True)

    #run_info = 'Pytorch_lightning_oversample2'
    #classifer2 = copy.deepcopy(classifier)
    #run_experiment(run_info, dm, classifier2)
    #del dm
    #del classifier2

    ########## Experiment 3 #########
    # trans = transforms.Compose([ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #          transforms.CenterCrop(224)])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                          num_workers=4, oversample=True, oversample_amt=.3)
    #
    # run_info = 'Pytorch_lightning_oversample3'
    # classifier3 = copy.deepcopy(classifier)
    # run_experiment(run_info, dm, classifier3)
    # del dm
    # del classifier3
    #
    # gc.collect()
    #
    ########## Experiment 4 #########
    #trans = transforms.Compose([ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #         transforms.CenterCrop(224)])
    #dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                         num_workers=4, oversample=True, oversample_amt=.5)

    #run_info = 'Pytorch_lightning_oversample5'
    #classifier4 = copy.deepcopy(classifier)
    #run_experiment(run_info, dm, classifier4)

    #del dm
    #del classifier4

    #gc.collect()

    ########## Experiment 5 #########
    # random flip
    #trans = transforms.Compose(
    #    [ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #     transforms.CenterCrop(224), transforms.RandomVerticalFlip()])
    #dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                        num_workers=4, oversample=True, oversample_amt=.5)

    #run_info = 'Pytorch_lightning_oversample5_flip'
    #classifier5 = copy.deepcopy(classifier)
    #run_experiment(run_info, dm, classifier5)

    #del dm
    #del classifier5

    #gc.collect()

    ########## Experiment 6 #########
    # tune all weights
    #trans = transforms.Compose(
    #    [ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #     transforms.CenterCrop(224), transforms.RandomVerticalFlip()])
    #dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                        num_workers=4, oversample=True, oversample_amt=.5)

    #vgg16 = models.vgg16_bn()
    # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    #print(vgg16.classifier[6].out_features)  # 1000

    # Newly created modules have require_grad=True by default
    #num_features = vgg16.classifier[6].in_features
    #features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    #features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    #vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    #print(vgg16)

    #classifier6 = TrachomaClassifier(vgg16)

    #run_info = 'Pytorch_lightning_oversample5_flip_allLayers'
    #run_experiment(run_info, dm, classifier6)

    #del dm
    #del classifier6

    #gc.collect()

    ########## Experiment 7 #########
    # tune all weights, optimizer monitoring recall
    # trans = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #      transforms.CenterCrop(224), transforms.RandomVerticalFlip()])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                         num_workers=4, oversample=True, oversample_amt=.5)
    #
    # vgg16 = models.vgg16_bn()
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier7 = TrachomaClassifier(vgg16, 'val_rec')
    #
    # run_info = 'Pytorch_lightning_oversample5_flip_allLayers_valRecall'
    # run_experiment(run_info, dm, classifier7)
    #
    # del dm
    # del classifier7
    #
    # gc.collect()

    ########## Experiment 8 #########
    # tune all weights, add weights to cross entropy loss
    # trans = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #      transforms.RandomVerticalFlip(), transforms.CenterCrop(224)])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms=trans, batch_size=5,
    #                         num_workers=4)
    #
    # tot_class1 = sum(dm.train_keys[:, 1])
    # tot_class0 = len(dm.train_keys) - tot_class1
    # weights = torch.tensor([tot_class0, tot_class1])
    # weights = weights / weights.sum()
    # weights = 1.0 / weights
    # weights = weights / weights.sum()
    # print('Loss Weights: ', weights)
    #
    # vgg16 = models.vgg16_bn()
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier8 = TrachomaClassifier(vgg16, weight=weights)
    #
    # run_info = 'Pytorch_lightning_flip_allLayers_weightedLoss'
    # run_experiment(run_info, dm, classifier8)
    #
    # del dm
    # del classifier8
    #
    # gc.collect()

    ########## Experiment 9 #########
    # tune all weights, oversample 0.5, tansforms: horizontal flip, rotation, perspective
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(), transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224), transforms.Normalize(data_stats['mean'], data_stats['std'])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1, batch_size=5,
    #                         num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg16 = models.vgg16_bn()
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier9 = TrachomaClassifier(vgg16)
    #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_allLayers'
    # run_experiment(run_info, dm, classifier9)
    #
    # del dm
    # del classifier9
    #
    # gc.collect()
    #
    # ########## Experiment 10 #########
    # # random weights/not pretrained, oversample 0.5, tansforms: horizontal flip, rotation, perspective
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(data_stats['mean'], data_stats['std']), transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(data_stats['mean'], data_stats['std'])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg16 = models.vgg16_bn(pretrained=False)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)

    # classifier10 = TrachomaClassifier(vgg16)
    #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_allLayers_randomWeights'
    # run_experiment(run_info, dm, classifier10)
    #
    # del dm
    # del classifier10
    #
    # gc.collect()

    # ########## Experiment 11 #########
    # #pretrained vgg16 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg16 = models.vgg16_bn(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier11 = TrachomaClassifier(vgg16)
    # #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_allLayers_pretrained_norm'
    # run_experiment(run_info, dm, classifier11)
    #
    # del dm
    # del classifier11
    #
    # gc.collect()

    ########## Experiment 12 #########
    # pretrained vgg16 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective, color jitter
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg16 = models.vgg16_bn(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier12 = TrachomaClassifier(vgg16)
    # #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_cJitter_allLayers_pretrained_norm_test'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # # ########## Experiment 13 #########
    # # # pretrained vgg16 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective, color jitter
    # # # definative not Trachoma - change threshold 0.2
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg16 = models.vgg16_bn(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier12 = TrachomaClassifier(vgg16, threshold=0.2)
    # #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_cJitter_allLayers_pretrained_norm_thresh2'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # # # ########## Experiment 14 #########
    # # # pretrained vgg16 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective, color jitter
    # # # definative not Trachoma - change threshold 0.8
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg16 = models.vgg16_bn(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier12 = TrachomaClassifier(vgg16, threshold=0.8)
    # #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_cJitter_allLayers_pretrained_norm_thresh8'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 15 #########
    # pretrained vgg16 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective, color jitter
    # graded vs ungraded
    # # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkeyWungrades.csv'
    #
    # trans_0 = transforms.Compose(
    #   [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #    [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #     transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)])),
    #     transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ICAPS', transforms_0=trans_0, transforms_1=trans_1,
    #                        batch_size=6, num_workers=4, oversample=True, mapp={'No TF': 0, 'TF': 0, 'Ungradeable': 1})
    #
    # vgg16 = models.vgg16_bn(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
    # print(vgg16)
    #
    # classifier12 = TrachomaClassifier(vgg16)
    #
    # run_info = 'Pytorch_lightning_pretrained_norm_ungraded_vgg2'
    # run_experiment(run_info, dm, classifier12, project='Trachoma_gradable')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # # ########## Experiment 16 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective, color jitter
    # # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # #features = list(res101.classifier.children())[:-1]  # Remove last layer
    # #features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_perspective_cJitter_allLayers_pretrained_norm_resnet2'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_0,
    #                         batch_size=5, num_workers=4, oversample=False)#, oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # #features = list(res101.classifier.children())[:-1]  # Remove last layer
    # #features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101, weight=torch.tensor([20]))
    # #
    # run_info = 'Pytorch_lightning_posweight20_pretrained_norm_resnet2'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    # #
    # # ########## Experiment 17 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation
    # # gradeable vs ungradeable
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'trachomagroundtruthkeyWungrades.csv'
    #
    # trans_0 = transforms.Compose(
    #   [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #    [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ICAPS', transforms_0=trans_0, transforms_1=trans_1,
    #                        batch_size=5, num_workers=4, oversample=True, mapp={'No TF': 0, 'TF': 0, 'Ungradeable': 1})
    #
    # res101 = models.resnet101(pretrained=True)
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotate_allLayers_pretrained_norm_resnet2'
    # run_experiment(run_info, dm, classifier12, project='Trachoma_gradable')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 18 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective, color jitter
    # # graded vs ungraded
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkeyWungrades.csv'
    #
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ICAPS', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, mapp={'No TF': 0, 'TF': 0, 'Ungradeable': 1})
    #
    # res101 = models.resnet101(pretrained=True)
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    #
    # run_info = 'Pytorch_lightning_flip_rotation_perspective_cJitter_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma_gradable')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 19 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation,  custom crop
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    #
    # run_info = 'Pytorch_lightning_flip_rotation_customCrop_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # # ########## Experiment 20 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation,  custom crop, YCrCb
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(False), ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(False), ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=False)
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    #
    # run_info = 'Pytorch_lightning_flip_rotation_customCrop_YCrCb_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 20 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation,  custom crop, YCrCb
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet34(pretrained=False)
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    #
    # run_info = 'Pytorch_lightning_oversample5_flip_rotation_customCrop_resnet34'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 21 #########
    # # pretrained resnet34 normalized to fit, tansforms: custom crop, Follicle Enhancment
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(), #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(), transforms.Resize(230), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, #transforms_1=trans_1,
    #                         batch_size=6, num_workers=4) #, oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet34(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_customCrop_follicleEnhance_resnet34'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # # ########## Experiment 22 #########
    # # pretrained resnet34 normalized to fit,tansforms: custom crop, Follicle Enhancment replace green layer
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(), transforms.Resize(230), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0,  # transforms_1=trans_1,
    #                         batch_size=6, num_workers=4)  # , oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet34(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_customCrop_follicleEnhanceReplaceGreen_resnet34'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # ########## Experiment 21 #########
    # pretrained resnet34 normalized to fit, oversample 0.5, tansforms: custom crop, Follicle Enhancment
    # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.Resize(230), transforms.CenterCrop(224)])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4 , oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet101(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhance_rotate_flip_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 22 #########
    # # pretrained resnet34 normalized to fit, oversample 0.5, tansforms: custom crop, Follicle Enhancment replace green layer
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(), transforms.Resize(230), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0,  # transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet34(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhanceReplaceGreen_resnet34'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    ########## Experiment 23 #########
    # pretrained resnet34 normalized to fit, oversample 0.5, tansforms: rotation, flip, custom crop, Follicle Enhancment replace green layer
    # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(), transforms.Resize(230), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224),])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet101(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhanceReplaceGreen_flip_rotate_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # # ########## Experiment 23 #########
    # # pretrained resnet34 normalized to fit, oversample 0.5, tansforms: rotation, flip, custom crop, Follicle Enhancment replace blue layer
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False, replace=2), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False, replace=2), ToTensor(), transforms.Resize(230),
    #      transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet101(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhanceReplaceBlue_flip_rotate_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # # ########## Experiment 23 #########
    # # pretrained resnet34 normalized to fit, oversample 0.5, tansforms: rotation, flip, perspective custom crop, Follicle Enhancment replace green layer
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(), transforms.Resize(230),
    #      transforms.RandomHorizontalFlip(), transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet101(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhanceReplaceGreen_flip_rotate_perspective_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()
    #
    # # ########## Experiment 24 #########
    # # resnet34 normalized to fit, oversample 0.5, tansforms: rotation, flip custom crop, Follicle Enhancment HSV
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(returnRGB=False), ToTensor(), transforms.Resize(230),
    #      transforms.RandomHorizontalFlip(), #transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet101(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhance_flip_rotate_HSV_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 25 #########
    # # resnet34 normalized to fit, oversample 0.5, tansforms: rotation, flip custom crop, Follicle Enhancment just s3
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(sonly=True), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(sonly=True), ToTensor(), transforms.Resize(230),
    #      transforms.RandomHorizontalFlip(), #transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res34 = models.resnet101(pretrained=False)
    # print(res34)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res34.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res34.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res34)
    #
    # classifier12 = TrachomaClassifier(res34)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhance_flip_rotate_s3_resnet101'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 26#########
    # # vgg11_bn normalized to fit, oversample 0.5, tansforms: rotation, flip custom crop, Follicle Enhancment
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(230),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [CustomCrop(), FollicleEnhance(), ToTensor(), transforms.Resize(230),
    #      transforms.RandomHorizontalFlip(),
    #      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg11 = models.vgg11_bn(pretrained=True)
    # print(vgg11)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg11.classifier[-1].in_features
    # features = list(vgg11.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # vgg11.classifier = nn.Sequential(*features)
    # print(vgg11)
    #
    # classifier12 = TrachomaClassifier(vgg11)
    #
    # run_info = 'Pytorch_lightning_oversample5_customCrop_follicleEnhance_flip_rotate_pretrained_vgg11_bn'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 27 #########
    # # vgg11_bn normalized to fit, oversample 0.5, tansforms: rotation, flip Follicle Enhancment
    # # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [FollicleEnhance(), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(226),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [FollicleEnhance(), ToTensor(), transforms.Resize(226),
    #      transforms.RandomHorizontalFlip(),
    #      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg11 = models.vgg11_bn(pretrained=False)
    # print(vgg11)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg11.classifier[-1].in_features
    # features = list(vgg11.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # vgg11.classifier = nn.Sequential(*features)
    # print(vgg11)
    #
    # classifier12 = TrachomaClassifier(vgg11)
    #
    # run_info = 'Pytorch_lightning_oversample5_follicleEnhance_flip_rotate_vgg11_bn'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 27 #########
    # vgg11_bn normalized to fit, oversample 0.5, tansforms: rotation, flip Follicle Enhancment HSV
    # TF vs NoTF
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
    #
    # trans_0 = transforms.Compose(
    #     [FollicleEnhance(returnRGB=False), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(226),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [FollicleEnhance(returnRGB=False), ToTensor(), transforms.Resize(226),
    #      transforms.RandomHorizontalFlip(),
    #      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # vgg11 = models.vgg11_bn(pretrained=False)
    # print(vgg11)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = vgg11.classifier[-1].in_features
    # features = list(vgg11.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # vgg11.classifier = nn.Sequential(*features)
    # print(vgg11)
    #
    # classifier12 = TrachomaClassifier(vgg11)
    #
    # run_info = 'Pytorch_lightning_oversample5_follicleEnhanceHSV_flip_rotate_vgg11_bn'
    # run_experiment(run_info, dm, classifier12, project='Trachoma')
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, pos_weight 40, tansforms: horizontal flip, rotation, perspective
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    #
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_0,
    #                         batch_size=6, num_workers=4, oversample=False)  # , oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101, weight=torch.tensor([20]))
    # #
    # run_info = 'Pytorch_lightning_consensus_posweight20_pretrained_norm_resnet101'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, pos_weight 40, tansforms: horizontal flip, rotation, perspective
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    #
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_0,
    #                         batch_size=6, num_workers=4, oversample=False)  # , oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101, weight=torch.tensor([40]))
    # #
    # run_info = 'Pytorch_lightning_consensus_posweight40_pretrained_norm_resnet101'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_consensus_oversample5_flop_rotate_perspective_pretrained_norm_resnet101'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, rotation, perspective accumulate batch 3
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(256),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_consensus_oversample5_flop_rotate_perspective_pretrained_norm_accumuate3batches_resnet101'
    # run_experiment(run_info, dm, classifier12, accum_batches=3)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, folicle enhance normalized
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [FollicleEnhance(), ToTensor(),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(226),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [FollicleEnhance(), ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize(226),
    #      transforms.RandomHorizontalFlip(),
    #      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_consensus_oversample5_follicleenhance_flip_rotate_pretrained_norm_resnet101'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, folicle enhance hsv normalized
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [FollicleEnhance(returnRGB=False), ToTensor(),
    #      transforms.Resize(226),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [FollicleEnhance(returnRGB=False), ToTensor(),
    #      transforms.Resize(226),
    #      transforms.RandomHorizontalFlip(),
    #      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5, normalize=True)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_consensus_oversample5_follicleenhancehsv_flip_rotate_resnet101'
    # run_experiment(run_info, dm, classifier12)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 17 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, folicle enhance hsv normalized accumulate 3 batches
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [FollicleEnhance(returnRGB=False), ToTensor(),
    #      transforms.Resize(226),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [FollicleEnhance(returnRGB=False), ToTensor(),
    #      transforms.Resize(226),
    #      transforms.RandomHorizontalFlip(),
    #      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5, normalize=True)
    #
    # res101 = models.resnet101(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_consensus_oversample5_follicleenhancehsv_flip_rotate_accumulate3batches_resnet101'
    # run_experiment(run_info, dm, classifier12, accum_batches=3)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # # ########## Experiment 17 #########
    # # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, folicle enhance normalized accumulate 3 batches
    # img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    # img_keys = '2300consensus8-2021.csv'
    #
    # trans_0 = transforms.Compose(
    #     [FollicleEnhance(), ToTensor(),
    #      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(226),
    #      transforms.CenterCrop(224)])
    # trans_1 = transforms.Compose(
    #     [FollicleEnhance(), ToTensor(), #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #      transforms.Resize(226),
    #      transforms.RandomHorizontalFlip(),
    #      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5, normalize=True)
    #
    # res101 = models.resnet101(pretrained=False)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)
    #
    # classifier12 = TrachomaClassifier(res101)
    # #
    # run_info = 'Pytorch_lightning_consensus_oversample5_follicleenhance_flip_rotate_norm_accum3batch_swa_resnet101'
    # run_experiment(run_info, dm, classifier12, accum_batches=3, swa=True)
    #
    # del dm
    # del classifier12
    #
    # gc.collect()

    # ########## Experiment 17 #########
    # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, folicle enhance normalized accumulate 3 batches
    img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    img_keys = '2300consensus8-2021.csv'
    # img_keys = 'TrachomaData/trachomagroundtruthkey.csv'

    trans_0 = transforms.Compose(
        [FollicleEnhance(), ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.Resize(226),
         transforms.CenterCrop(224)])
    trans_1 = transforms.Compose(
        [FollicleEnhance(), ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.Resize(226),
         transforms.RandomHorizontalFlip(),
         # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
         transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
    # dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
    #                         batch_size=5, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    # res101 = models.resnet18(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(res101)  # 1000
    #
    # # Newly created modules have require_grad=True by default
    # num_features = res101.fc.in_features
    # # features = list(res101.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    # res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    # print(res101)

    # for i in [4]:
    #     for j in [5]:
    #         if i == 4 and j == 3:
    #             continue
    #         dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ICAPS', transforms_0=trans_0,
    #                                 transforms_1=trans_1,
    #                                 batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)
    #
    #         res101 = models.resnet18(pretrained=True)
    #         # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    #         print(res101)  # 1000
    #
    #         # Newly created modules have require_grad=True by default
    #         num_features = res101.fc.in_features
    #         # features = list(res101.classifier.children())[:-1]  # Remove last layer
    #         # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
    #         res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
    #         print(res101)
    #         classifier12 = TrachomaClassifier(res101, weight=torch.tensor([i]))
    #         #
    #         run_info = 'Pytorch_lightning_ICAPS_oversample5_posweight{}_follicleenhance_flip_rotate_norm_pretrained_accum{}batch_resnet18'.format(i, j)
    #         run_experiment(run_info, dm, classifier12, accum_batches=j)
    #         del classifier12
    #         del dm
    #         gc.collect()

    for i in [4]:
        for j in [5]:
            if i == 4 and j == 3:
                continue
            dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0,
                                    transforms_1=trans_1,
                                    batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5)

            res101 = models.resnet101(pretrained=True)
            # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
            print(res101)  # 1000

            # Newly created modules have require_grad=True by default
            num_features = res101.fc.in_features
            # features = list(res101.classifier.children())[:-1]  # Remove last layer
            # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
            res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
            print(res101)
            classifier12 = TrachomaClassifier(res101, weight=torch.tensor([i]))
            #
            run_info = 'Pytorch_lightning_consensus_oversample5_posweight{}_follicleenhance_flip_rotate_norm_pretrained_accum{}batch_resnet101'.format(i, j)
            run_experiment(run_info, dm, classifier12, accum_batches=j)
            del classifier12
            del dm
            gc.collect()









