import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import pickle
import pdb
import os


def accuracy(output, target, is_train=True, state=[0, 0], save=None):
    """
    train:
    output: [num_bag, r]
    target: [num_bag]
    test:
    output: [num_bag, r, r]
    target: [num_bag, r]
    """
    with torch.no_grad():
        if state is None:
            state = [0, 0]
        if is_train:
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = torch.sum(pred == target).item()
            state[0] += correct
            state[1] += len(target)
        else:
            pred = torch.argmax(output, dim=2)
            pred = torch.eq(pred, torch.arange(target.shape[1]).unsqueeze(0))
            correct = torch.sum(pred.int() * target.int()).item()
            state[0] += correct
            state[1] += torch.sum(target).item()
        return state[0] / state[1] if state[1] != 0 else 0, state


def non_na_accuracy(output, target, is_train=True, state=[0, 0], save=None):
    with torch.no_grad():
        if state is None:
            state = [0, 0]
        if is_train:
            non_na_index = ~target.eq(0)
            output = torch.masked_select(output, non_na_index.unsqueeze(1)).reshape(
                -1, output.shape[1]
            )
            target = torch.masked_select(target, non_na_index)
            if len(target) != 0:
                pred = torch.argmax(output, dim=1)
                assert pred.shape[0] == len(target)
                state[0] += torch.sum(pred == target).item()
                state[1] += len(target)
        else:
            pred = torch.argmax(output, dim=2)
            pred = torch.eq(pred, torch.arange(target.shape[1]).unsqueeze(0))
            state[0] += torch.sum(pred.double()[:, 1:] * target[:, 1:]).item()
            state[1] += torch.sum(target[:, 1:]).item()
        return state[0] / state[1] if state[1] != 0 else 0, state


def auc(output, target, is_train=False, state=None, save=None):
    with torch.no_grad():
        output = torch.diagonal(F.softmax(output, 2), offset=0, dim1=1, dim2=2)
        result = {
            "output": output[:, 1:].cpu().numpy(),
            "target": target[:, 1:].cpu().numpy(),
        }
        output = torch.flatten(output[:, 1:])
        target = torch.flatten(target[:, 1:])
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            target.cpu(), output.cpu()
        )
        auc = sklearn.metrics.auc(x=recall, y=precision)
        if save is not None:
            with open(os.path.join(save, "test_results.pickle"), "wb") as f:
                result.update({"recall": recall, "precision": precision})
                pickle.dump(result, f)
            p_at_r = [0, 0, 0]
            for p, r in zip(precision[::-1], recall[::-1]):
                if r >= 0.1 and p_at_r[0] == 0:
                    p_at_r[0] = p
                if r >= 0.2 and p_at_r[1] == 0:
                    p_at_r[1] = p
                if r >= 0.3 and p_at_r[2] == 0:
                    p_at_r[2] = p
            print(
                "P@0.1: %.4f\nP@0.2: %.4f\nP@0.3:%.4f"%(p_at_r[0], p_at_r[1], p_at_r[2])
            )

        return auc, None

