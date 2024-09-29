import argparse
import json
import os
import os.path as osp
import random
import shutil
import statistics
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
# from add_noise_to_data.random_noise import RandomNoiseAdder
from dataset.shapenet import Shapes3dDataset
from models import  PointNetAE
from tqdm import tqdm
from utils import load_model_for_evaluation
import ot

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

def compute_true_Wasserstein(X,Y,alpha=0.5):
    X1,X2 = X[:,:-3],X[:,-3:]
    Y1, Y2 = Y[:, :-3], Y[:, -3:]
    M1= ot.dist(X1.cpu().detach().numpy(), Y1.cpu().detach().numpy())
    M2 = np.sqrt(ot.dist(X2.cpu().detach().numpy(), Y2.cpu().detach().numpy()))
    M2=(2*np.arcsin(M2/2))**2
    M=alpha*M1+(1-alpha)*M2
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)**0.5
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--logdir", type=str, help="folder contains weights")
    parser.add_argument("--data_path", default="data/shapenet_psr", type=str, help="path to data")
    parser.add_argument("--path", default="epoch_20.pth", type=str, help="path to data")
    args = parser.parse_args()
    config = args.config
    logdir = args.logdir
    data_path = args.data_path
    path = args.path
    args = json.load(open(config))
    args["model_path"]=path
    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    # save_results folder
    save_folder = osp.join(logdir, args["save_folder"])
    if not osp.exists(save_folder):
        try:
            os.makedirs(save_folder)
            print(">Save_folder was created successfully at:", save_folder)
        except:
            pass
    print("You have 5s to check the hyperparameters below.")
    print(args)
    time.sleep(5)

    # device
    device = torch.device(args["device"])


    # dataloader
    if args["dataset_type"] == "shapenet":
        print(data_path)
        dset = Shapes3dDataset(dataset_folder=data_path, num_points=args["num_points"], split="test")
    else:
        raise ValueError("Unknown dataset type.")

    loader = data.DataLoader(
        dset,
        batch_size=1,
        pin_memory=True,
        num_workers=args["num_workers"],
        shuffle=args["shuffle"],
        worker_init_fn=seed_worker,
    )

    # distance

    # architecture
    if args["architecture"] == "pointnet":
        ae = PointNetAE(
            args["embedding_size"],
            args["input_channels"],
            args["input_channels"],
            args["num_points"],
            args["normalize"],
        ).to(device)


    else:
        raise ValueError("Unknown architecture.")

    try:
        ae = load_model_for_evaluation(ae, args["model_path"])
    except:
        try:
            ae = load_model_for_evaluation(ae, osp.join(logdir, args["model_path"]))
        except:
            in_dic = {"key": "autoencoder"}
            ae = load_model_for_evaluation(ae, osp.join(logdir, args["model_path"]), **in_dic)

    emd_list = []

    ae.eval()
    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader)):
            batch = batch[0].float().to(device)

            # inp = batch.detach().clone()

            if args["add_noise"]:
                args["test_origin"] = True

            try:
                reconstruction = ae.decode(ae.encode(batch))
            except:
                latent, reconstruction = ae(batch)

            emd_list.append(compute_true_Wasserstein(reconstruction.view(-1, 6), batch.view(-1, 6)))
    # report
    np.savetxt(save_folder + 'wd_{}.txt'.format(args["model_path"]), [np.mean(emd_list)])

    test_log = osp.join(save_folder, "reconstruction_test.log")
    _log = "{}| stddev {}| mean {}\n"

    plt.hist(emd_list, density=False, bins=args["bin"])
    plt.ylabel("Frequency")
    plt.xlabel("EMD value")
    save_path = osp.join(save_folder, "emd.eps")
    plt.savefig(save_path)
    plt.clf()

    log = _log.format("emd", statistics.stdev(emd_list), statistics.mean(emd_list))
    with open(test_log, "a") as fp:
        fp.write(log)
        print(log)


if __name__ == "__main__":
    main()
