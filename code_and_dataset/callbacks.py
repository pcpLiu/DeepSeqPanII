import sys

import torch

class EarlyStopCallBack(object):
    """
    Early stop
    """

    def __init__(self, patience=5, delta=0, initial_min_loss=0):
        self.patience = patience
        self.delta = delta
        self.min_loss = initial_min_loss
        self.last_update_epoch = -1

    def check(self, epoch, new_loss):
        if epoch == 0:
            self.last_update_epoch = 0
            self.min_loss = new_loss
            return False

        if new_loss <= self.min_loss - self.delta:
            # loss decrease, keep going
            self.last_update_epoch = epoch
            self.min_loss = new_loss
            return False
        else:
            if epoch < self.last_update_epoch + self.patience:
                # in patience, fine
                return False
            else:
                print("    - [EarlyStopCallBack]: Loss did not imporve in {} epochs. Stops training. ".format(self.patience))
                return True


class ModelCheckPointCallBack(object):
    """
    Model checking and save best model accordign to loss value.
    Save parameters ONLY.
    """

    def __init__(self, model, save_path, period=1, delta=0, initial_min_loss=0):
        self.model = model
        self.save_path = save_path
        self.period = period
        self.delta = delta
        self.min_loss = initial_min_loss
        self.last_update_epoch = -1

    def check(self, epoch, new_loss, save_path=None):
        if epoch == 0:
            self.last_update_epoch = 0
            self.min_loss = new_loss
            if save_path is None:
                    save_path = self.save_path
            print("    - [ModelCheckPointCallBack]: initial saving. Save model states to [{}].".format(save_path))
            torch.save(self.model.state_dict(), save_path)
            return

        # in period
        if epoch < self.last_update_epoch + self.period:
            return
        else:
            if new_loss <= self.min_loss - self.delta:
                if save_path is None:
                    save_path = self.save_path

                print("    - [ModelCheckPointCallBack]: Loss improved from {0:0.4f} to {1:0.4f}. Save model states to [{2}].".format(
                    self.min_loss, new_loss, save_path))
                torch.save(self.model.state_dict(), save_path)

                # update
                self.min_loss = new_loss
                self.last_update_epoch = epoch
