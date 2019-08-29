import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
from data_loader.data_loaders import load_word_dict, load_rel_dict
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from trainer import assign_to_device
import numpy as np
import pdb
import pickle


def main(config, args):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # setup data_loader instances
    word2id, word2vec = load_word_dict(
        os.path.join(args.input, "word_vec.json"), anony=False
    )
    rel2id = load_rel_dict(os.path.join(args.input, "rel2id.json"))

    # build model architecture
    model = get_instance(
        module_arch, "arch", config, word_vec_mat=word2vec, relation_num=len(rel2id)
    )
    print(model)

    # get function handles of metrics
    metric_fns = [getattr(module_metric, met) for met in config["eval_metrics"]]

    # load state dict
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    save = args.save
    # setup data_loader instances
    data_dir = args.input
    test_data_loader = module_data.BaseNytLoader(
        data_dir,
        word2id,
        rel2id,
        120,
        2000,
        validation_split=0,
        mode="test",
        src="test",
        method=config["data_loader"]["args"]["method"]
        if args.method is None
        else args.method,
        batch_type=1,
        select=0,
        filtering_mode=0,
        shuffle=False,
        num_workers=1,
        anonymization=False,
    )
    print("%d entities pairs in total" % len(test_data_loader.subset))

    total_output = []
    total_target = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_data_loader)):
            data, target = (
                assign_to_device(data, device),
                assign_to_device(target, device),
            )
            output = model(data, is_train=False)
            total_output.append(output)
            total_target.append(target)
        total_output = torch.cat(total_output)
        total_target = torch.cat(total_target)
        total_metrics = np.zeros(len(metric_fns))
        for i, metric in enumerate(metric_fns):
            total_metrics[i], _ = metric(
                total_output, total_target, is_train=False, save=save
            )
    log = {}
    log.update(
        {
            met.__name__: total_metrics[i].item()
            for i, met in enumerate(metric_fns)
        }
    )
    for metric_name, metric_value in log.items():
        print("%s: %.4f"%(metric_name, metric_value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")

    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        required=True,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        required=True,
        type=str,
        help="input data for evauation"
    )
    parser.add_argument(
        "-s", "--save", default=None, type=str, help="path to save evaluation results"
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        type=int,
        help="method for RE, 0 for BASE, 1 for MERGE, 2 for REDS2"
    )

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.resume:
        config = torch.load(args.resume)["config"]

    main(config, args)
