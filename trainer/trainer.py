import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs, string_classes, int_classes

# from torchvision.utils import make_grid
from base import BaseTrainer
import pdb


def assign_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, container_abcs.Mapping):
        new_data = {}
        for key in data:
            new_data[key] = assign_to_device(data[key], device)
        return new_data
    elif isinstance(data, int_classes):
        return data
    elif isinstance(data, float):
        return data
    elif isinstance(data, string_classes):
        return data
    else:
        return data


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        model,
        loss,
        train_metrics,
        eval_metrics,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        weight=None,
    ):
        super(Trainer, self).__init__(
            model, loss, train_metrics, eval_metrics, optimizer, config
        )
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(len(data_loader) / 5)
        self.weight = weight

    def _train_metrics(self, output, target, states=None):
        if states is None:
            states = [None] * len(self.train_metrics)
        acc_metrics = np.zeros(len(self.train_metrics))
        # pdb.set_trace()
        for i, metric in enumerate(self.train_metrics):
            result, state = metric(output, target, state=states[i])
            acc_metrics[i] = result
            states[i] = state
            self.writer.add_scalar("{}".format(metric.__name__), acc_metrics[i])
        return acc_metrics, states

    def _eval_metrics(self, output, target, states=None):
        if states is None:
            states = [None] * len(self.eval_metrics)
        acc_metrics = np.zeros(len(self.eval_metrics))
        for i, metric in enumerate(self.eval_metrics):
            result, state = metric(output, target, state=states[i], is_train=False)
            acc_metrics[i] = result
            states[i] = state
            self.writer.add_scalar("{}".format(metric.__name__), acc_metrics[i])
        return acc_metrics, states

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        states = [None] * len(self.train_metrics)
        # pdb.set_trace()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # pdb.set_trace()
            # if batch_idx == 0:
            #     print(target)
            data, target = (
                assign_to_device(data, self.device),
                assign_to_device(target, self.device),
            )
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target, self.weight)
            loss.backward()
            # pdb.set_trace()
            nn.utils.clip_grad_value_(filter(lambda p: p.requires_grad, self.model.parameters()), 1)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar("loss", loss.item())
            total_loss += loss.item()
            train_metrics, states = self._train_metrics(output, target, states)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * self.data_loader.batch_size if self.data_loader.batch_size!=None else 'na',
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item(),
                    )
                )
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            "loss": total_loss / len(self.data_loader),
            "metrics": train_metrics.tolist(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_output = []
        total_target = []
        total_val_loss = 0
        with torch.no_grad():
            # pdb.set_trace()
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = (
                    assign_to_device(data, self.device),
                    assign_to_device(target, self.device),
                )
                # print(data['all_relations'][:,1:].nonzero())
                output = self.model(data, is_train=False)
                loss = self.loss(output, target, is_train=False, weight=self.weight)
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.writer.add_scalar("loss", loss.item())
                total_val_loss += loss.item()
                total_output.append(output)
                total_target.append(target)
            total_output = torch.cat(total_output)
            total_target = torch.cat(total_target)
            self.writer.set_step(epoch, "valid")
            val_metrics, _ = self._eval_metrics(total_output, total_target)
            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return {
            "val_loss": total_val_loss / len(self.valid_data_loader),
            "val_metrics": val_metrics.tolist(),
        }
