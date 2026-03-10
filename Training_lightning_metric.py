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
from pytorch_metric_learning import losses, miners
import torchmetrics
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from MetricModels import Resnet18

from DataLoader_lightning import TrachomaDataModule, ToTensor, CustomCrop, FollicleEnhance


class TrachomaClassifier(pl.LightningModule):
    def __init__(self, model, optimizer_metric='train_loss', weight=None, threshold=0.5, classify=False):
        super().__init__()
        # self.features = {}

        # self.model = MidGetter(model, return_layers=return_layers, keep_output=True)
        self.model = model
        self.classify = classify
        # self.model.layer3.register_forward_hook(self.get_embeddings('emb'))

        self.optimizer_metric = optimizer_metric
        # self.loss = nn.CrossEntropyLoss(weight=weight)
        self.loss_class = nn.BCEWithLogitsLoss(pos_weight=weight)
        self.loss_metric = losses.TripletMarginLoss()
        self.miner = miners.MultiSimilarityMiner()

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

        # outputs = self.model(images).squeeze(1)
        # loss = self.loss(outputs, targets.float())

        if self.classify is False:
            embeddings = self.model(images, full=self.classify)
            hard_pairs = self.miner(embeddings, targets)
            loss = self.loss_metric(embeddings, targets, hard_pairs)
            outputs = None
        else:
            output = self.model(images)
            outputs = output.squeeze(1)
            loss = self.loss_class(outputs, targets.float())

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def training_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['loss']

        if batch_parts['outputs'] is not None:
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

        output = self.model(images)

        outputs = output.squeeze(1)
        loss = self.loss_class(outputs, targets.float())

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

        output = self.model(images)

        outputs = output.squeeze(1)
        loss = self.loss_class(outputs, targets.float())

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

    def _configure_optim_metric(self):
        # return optimizers and schedulers for pre-training
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True, factor=0.1, cooldown=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': self.optimizer_metric,
            },
        }

    def _configure_optim_classify(self):
        # return optimizers and scheduler for fine-tine
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True, factor=0.1, cooldown=3)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': self.optimizer_metric,
                },
            }

    def configure_optimizers(self):
        if self.classify is False:
            return self._configure_optim_metric()
        elif self.classify:
            return self._configure_optim_classify()

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #
    #     if self.optimizer_metric == 'val_loss':
    #         mode = 'min'
    #     else:
    #         mode = 'max'
    #
    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': ReduceLROnPlateau(optimizer, mode=mode, patience=3, verbose=True, factor=0.1, cooldown=3),
    #             'interval': 'epoch',
    #             'frequency': 1,
    #             'monitor': self.optimizer_metric,
    #         },
    #     }


def run_experiment(run_info, dataloader, model, project='Trachoma', accum_batches=1, swa=False):
    print('Running Experiment: ', run_info)

    # train
    wandb_logger = WandbLogger(project=project, name=run_info)
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath='Checkpoints/{}'.format(run_info), save_last=True, save_top_k=1, mode='min', monitor='val_loss')
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=2, log_every_n_steps=20, logger=wandb_logger, max_epochs=40, #auto_lr_find=True,
                          default_root_dir='Checkpoints', accelerator='ddp', callbacks=[checkpoint_callback], accumulate_grad_batches=accum_batches, stochastic_weight_avg=swa)
        trainer.fit(model, dataloader)

        del dataloader

        trans_0 = transforms.Compose(
            [ToTensor(),
             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             transforms.Resize(226),
             transforms.CenterCrop(224)])
        trans_1 = transforms.Compose(
            [ToTensor(),  # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             transforms.Resize(226),
             transforms.RandomHorizontalFlip(),
             # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
             transforms.RandomRotation(10), transforms.CenterCrop(224), ])
        dataloader = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_1,
                                        transforms_1=trans_1,
                                        batch_size=8, num_workers=2, oversample=True, oversample_amt=0.5,
                                        normalize=True)

        # train metric larger image size
        path = 'Checkpoints/{}/last.ckpt'.format(run_info)
        try:
            # model = TrachomaClassifier(Resnet18(), cla)
            model = TrachomaClassifier.load_from_checkpoint(path, model=Resnet18(), classify=False, strict=True)
        except:
            state_dict = torch.load(path)
            state_dict2 = {'model.' + k: v for k, v in state_dict['state_dict'].items()}
            state_dict['state_dict'] = state_dict2
            torch.save(state_dict, path)
            model = TrachomaClassifier.load_from_checkpoint(path, model=Resnet18(), classify=False, strict=True)


        trainer = pl.Trainer(gpus=2, log_every_n_steps=20, logger=wandb_logger, max_epochs=80,  # auto_lr_find=True,
                             default_root_dir='Checkpoints', accelerator='ddp',
                             callbacks=[checkpoint_callback],
                             accumulate_grad_batches=accum_batches, stochastic_weight_avg=swa, resume_from_checkpoint=path)
        trainer.fit(model, dataloader)

        # train classifier
        path = 'Checkpoints/{}/last.ckpt'.format(run_info)
        try:
            model = TrachomaClassifier.load_from_checkpoint(path, model=Resnet18(), classify=True, strict=True)
        except:
            state_dict = torch.load(path)
            state_dict2 = {'model.' + k: v for k, v in state_dict['state_dict'].items()}
            state_dict['state_dict'] = state_dict2
            torch.save(state_dict, path)
            model = TrachomaClassifier.load_from_checkpoint(path, model=Resnet18(), classify=True, strict=True)

        trainer = pl.Trainer(gpus=2, log_every_n_steps=20, logger=wandb_logger, max_epochs=600,  # auto_lr_find=True,
                             default_root_dir='Checkpoints', accelerator='ddp',
                             callbacks=[checkpoint_callback, early_stop_callback],
                             accumulate_grad_batches=accum_batches,
                             stochastic_weight_avg=swa, resume_from_checkpoint=path)
        trainer.fit(model, dataloader)

    else:
        trainer = pl.Trainer(num_processes=1, log_every_n_steps=20, logger=wandb_logger, max_epochs=75, default_root_dir='Checkpoints', callbacks=[checkpoint_callback])
        trainer.fit(model, dataloader)


    # test
    # trainer.test()

    wandb.finish()

    del wandb_logger
    del trainer
    del model
    del dataloader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':

    # pretrained resnet101 normalized to fit, oversample 0.5, tansforms: horizontal flip, folicle enhance normalized accumulate 3 batches
    img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
    # img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
    img_keys = '2300consensus8-2021.csv'
    # img_keys = 'trachomagroundtruthkey.csv'

    trans_0 = transforms.Compose(
        [ToTensor(),
         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.Resize(114),
         transforms.CenterCrop(112)])
    trans_1 = transforms.Compose(
        [ToTensor(), #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.Resize(114),
         transforms.RandomHorizontalFlip(),
         # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
         transforms.RandomRotation(10), transforms.CenterCrop(112), ])
    dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_1, transforms_1=trans_1,
                            batch_size=8, num_workers=2, oversample=True, oversample_amt=0.5, normalize=True)

    res18 = Resnet18()

    classifier12 = TrachomaClassifier(res18)
    #
    run_info = 'Pytorch_lightning_consensus_oversample5_rotate_norm_resnet18_metric_112_224_resetLearning_1'
    run_experiment(run_info, dm, classifier12)

    del dm
    del classifier12

    gc.collect()

