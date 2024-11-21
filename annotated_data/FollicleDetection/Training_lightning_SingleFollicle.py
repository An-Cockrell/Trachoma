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

from DataLoader_lightning_SingleFollicle import TrachomaDataModule, ToTensor


class TrachomaGradableArea(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.loss = BCEDiceLoss(eps=1.0, activation='softmax2d')

        # metrics

        # self.train_precision = torchmetrics.Precision(multiclass=False, num_classes=1, threshold=threshold)
        # self.val_precision = torchmetrics.Precision(multiclass=False, num_classes=1, threshold=threshold)
        # self.test_precision = torchmetrics.Precision(multiclass=False, num_classes=1, threshold=threshold)
        #
        # self.train_recall = torchmetrics.Recall(multiclass=False, num_classes=1, threshold=threshold)
        # self.val_recall = torchmetrics.Recall(multiclass=False, num_classes=1, threshold=threshold)
        # self.test_recall = torchmetrics.Recall(multiclass=False, num_classes=1, threshold=threshold)
        #
        # self.train_f1 = torchmetrics.F1(multiclass=False, num_classes=1, threshold=threshold)
        # self.val_f1 = torchmetrics.F1(multiclass=False, num_classes=1, threshold=threshold)
        # self.test_f1 = torchmetrics.F1(multiclass=False, num_classes=1, threshold=threshold)

    def forward(self, x):
        classification = self.model(x)
        return classification

    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label']

        outputs = self.model(images)['out'].squeeze()
        loss = self.loss(outputs, targets)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def training_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['loss']
        outputs = batch_parts['outputs']
        targets = batch_parts['targets']

        # # log metrics
        # self.train_acc(outputs, targets)
        # self.train_precision(outputs, targets)
        # self.train_recall(outputs, targets)
        # self.train_f1(outputs, targets)
        #
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        # self.log('train_pre', self.train_precision, on_step=True, on_epoch=True)
        # self.log('train_rec', self.train_recall, on_step=True, on_epoch=True)
        # self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)

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

        outputs = self.model(images)['out'].squeeze()
        loss = self.loss(outputs, targets)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def validation_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['loss']
        outputs = batch_parts['outputs']
        targets = batch_parts['targets']

        # log metrics
        # self.val_acc(outputs, targets)
        # self.val_precision(outputs, targets)
        # self.val_recall(outputs, targets)
        # self.val_f1(outputs, targets)
        #
        # self.log('val_acc', self.val_acc)#, on_step=True, on_epoch=True)
        # self.log('val_pre', self.val_precision)#, on_step=True, on_epoch=True)
        # self.log('val_rec', self.val_recall)#, on_step=True, on_epoch=True)
        # self.log('val_f1', self.val_f1) #, on_step=True, on_epoch=True)

        # do something with both outputs
        loss = torch.sum(losses) / torch.numel(losses)
        self.log('val_loss', loss)

        #print(outputs, targets)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label']

        outputs = self.model(images)['out'].squeeze()
        loss = self.loss(outputs, targets)

        return {'loss': loss, 'outputs': outputs, 'targets': targets}

    def test_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['loss']
        outputs = batch_parts['outputs']
        targets = batch_parts['targets']

        # log metrics
        # self.test_acc(outputs, targets)
        # self.test_precision(outputs, targets)
        # self.test_recall(outputs, targets)
        # self.test_f1(outputs, targets)
        #
        # self.log('test_acc', self.test_acc) #, on_step=True, on_epoch=True)
        # self.log('test_pre', self.test_precision) #, on_step=True, on_epoch=True)
        # self.log('test_rec', self.test_recall) #, on_step=True, on_epoch=True)
        # self.log('test_f1', self.test_f1)#, on_step=True, on_epoch=True)

        # do something with both outputs
        loss = torch.sum(losses) / torch.numel(losses)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
            },
        }


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1.,
                           eps=self.eps, threshold=None,
                           activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid', lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = nn.BCELoss(reduction='mean')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)



def run_experiment(run_info, dataloader, model, project='Trachoma_GradeableArea', accum_batches=1, swa=False):
    print('Running Experiment: ', run_info)

    # train
    wandb_logger = WandbLogger(project=project, name=run_info)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath='Checkpoints/{}'.format(run_info), save_last=True, save_top_k=1, mode='min', monitor='val_loss')
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, log_every_n_steps=20, logger=wandb_logger, max_epochs=50,
                          default_root_dir='Checkpoints', accelerator='cuda', callbacks=[early_stop_callback, checkpoint_callback], accumulate_grad_batches=accum_batches)
    else:
        trainer = pl.Trainer(num_processes=1, log_every_n_steps=20, logger=wandb_logger, max_epochs=50, default_root_dir='Checkpoints', callbacks=[early_stop_callback, checkpoint_callback])
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
    img_dir = './singleFollicleImages/Images'
    mask_dir = './singleFollicleImages/Masks'

    trans_0 = transforms.Compose(
        [ToTensor(),
         transforms.Resize(110),
         transforms.CenterCrop(100)])
    trans_1 = transforms.Compose(
        [ToTensor(), transforms.Resize(110),
         transforms.RandomApply(nn.ModuleList(
             [transforms.RandomVerticalFlip(.5), transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(90),
              transforms.RandomPerspective(0.3)])), transforms.CenterCrop(100)
         ])

    dm = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0, transforms_1=trans_1,
                            batch_size=6, num_workers=1, oversample=True, oversample_amt=10, normalize=True)

    fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
    print(fcn)
    segModel = TrachomaGradableArea(fcn)
    run_info = 'Segmentation_Test_SingleFollicle_100'
    run_experiment(run_info, dm, segModel)