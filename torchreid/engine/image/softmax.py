from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss, AMSoftmaxLoss
from torchreid.utils.torchtools import get_model_attr

from ..engine import Engine


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        loss='softmax',
        conf_penalty=0.,
        margin=0.35,
        scale=30,
        pr_product=False,
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        if loss == 'softmax':
            self.criterion = CrossEntropyLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                label_smooth=label_smooth
            )
        elif loss == 'am_softmax':
            self.criterion = AMSoftmaxLoss(label_smooth=label_smooth,
                                           use_gpu=self.use_gpu, m=margin,
                                           s=scale, pr_product=pr_product,
                                           conf_penalty=conf_penalty)

        self.num_attrs = get_model_attr(self.model, 'attrs_num')
        if self.num_attrs:
            self.attr_criterion = CrossEntropyLoss(
                num_classes=self.num_attrs,
                use_gpu=self.use_gpu,
                label_smooth=label_smooth
            )


    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs = self.model(imgs)
        if self.num_attrs:
            loss = self.compute_loss(self.criterion, outputs[0], pids)
            attr_loss = 0.1 * self.compute_loss(self.attr_criterion, outputs[1], data['attr'])
            loss += attr_loss
        else:
            loss = self.compute_loss(self.criterion, outputs, pids)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }
        if self.num_attrs:
            loss_summary['attr_loss'] = attr_loss.item()

        return loss_summary
