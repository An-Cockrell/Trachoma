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

from DataLoader_lightning_mData_synth import TrachomaDataModule, ToTensor, CustomCrop, FollicleEnhance


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
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath='Checkpoints/{}'.format(run_info), save_last=True, save_top_k=1, mode='min', monitor='val_loss')
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, log_every_n_steps=20, logger=wandb_logger, max_epochs=50,
                          default_root_dir='Checkpoints', accelerator='ddp', callbacks=[early_stop_callback, checkpoint_callback], accumulate_grad_batches=accum_batches, stochastic_weight_avg=swa)
    else:
        trainer = pl.Trainer(num_processes=1, log_every_n_steps=20, logger=wandb_logger, max_epochs=50, default_root_dir='Checkpoints', callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, dataloader)

    # test
    trainer.test(ckpt_path='best', dataloaders=dataloader)

    wandb.finish()

    del wandb_logger
    del trainer
    del model
    del dataloader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    img_dir = 'm'
    img_keys = 'm/tfti.csv'

    img_dir_tf = 'synthetic_TF_images'
    img_dir_non_tf = 'synthetic_NonTF_images'

    trans_0 = transforms.Compose(
        [FollicleEnhance(), ToTensor(),
         transforms.Normalize(mean=[0.4324, 0.2903, 0.2679], std=[0.1566, 0.1189, 0.1178]),
         transforms.Resize(226),
         transforms.CenterCrop(224)])

    trans_1 = transforms.Compose(
        [FollicleEnhance(), ToTensor(),
         transforms.Normalize(mean=[0.4324, 0.2903, 0.2679], std=[0.1566, 0.1189, 0.1178]), transforms.Resize(226), transforms.RandomHorizontalFlip(),
         transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
         transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224)])

    for dp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        dm = TrachomaDataModule(img_dir, img_keys, img_dir_tf, img_dir_non_tf, transforms_0=trans_0, transforms_1=trans_0,
                            batch_size=10, num_workers=4, oversample=False, split=True, synthDataPercent=dp)

        res101 = models.resnet101(pretrained=False)
        # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
        # print(res101)  # 1000
        #
        # Newly created modules have require_grad=True by default
        num_features = res101.fc.in_features
        # features = list(res101.classifier.children())[:-1]  # Remove last layer
        # features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
        res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
        print(res101)
        #
        classifier12 = TrachomaClassifier(res101)
        # #
        run_info = 'Pytorch_lightning_follicleEnhance_flip_rotate_perspective_resnet101_m_synth_dp{}'.format(str(dp))
        run_experiment(run_info, dm, classifier12)

        del dm
        del classifier12

        gc.collect()
