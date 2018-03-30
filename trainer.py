from tqdm import tqdm
from utils import AverageMeter
import logging

logger = logging.getLogger('OCR')


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for training.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.stop_training = False

    def train(self, train_loader, val_loader, start_epoch=0, epochs=200, callbacks=[]):
        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                return
            train_loss, train_acc = self.train_one_epoch(epoch, train_loader, callbacks=callbacks)
            val_loss, val_acc = self.validate(epoch, val_loader)

            msg = "train loss: {:.3f} - train acc: {:.3f}  val loss: {:.3f} - val acc: {:.3f}"
            logger.info(msg.format(train_loss, train_acc, val_loss, val_acc))

            for cbk in callbacks:
                cbk.on_epoch_end(epoch, {'val_loss': val_loss, 'val_acc': val_acc})

    def train_one_epoch(self, epoch, train_loader, callbacks=[]):
        """
        Train the model for 1 epoch of the training set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(tqdm(train_loader, unit='batch', desc='Epoch {:>3}'.format(epoch))):
            metric = self.model.forward(x, y, is_training=True)
            loss = metric['loss']
            acc = metric['acc']

            losses.update(loss.data[0], x.size()[0])
            accs.update(acc.data[0], x.size()[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for cbk in callbacks:
                cbk.on_batch_end(epoch, i, logs=metric)

        return losses.avg, accs.avg

    def validate(self, epoch, val_loader):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(val_loader):
            # metric = self.model.forward(x, y, is_training=False)
            metric = self.model.forward(x, y)
            loss = metric['loss']
            acc = metric['acc']

            losses.update(loss.data[0], x.size()[0])
            accs.update(acc.data[0], x.size()[0])

        return losses.avg, accs.avg

    def test(self, test_loader, best=True):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        # load the best checkpoint
        self.load_checkpoint(best=best)

        accs = AverageMeter()

        for i, (x, y) in enumerate(test_loader):
            metric = self.model.forward(x, y)
            acc = metric['acc']

            accs.update(acc.data[0], x.size()[0])

        logger.info('Test Acc: {}/{} ({:.2f}%)'.format(accs.sum, accs.n, accs.avg))
