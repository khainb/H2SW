import argparse
import json
import os
import os.path as osp
import random
import shutil
import time
import datetime

import numpy as np
import torch
from dataset import Shapes3dDataset
from logger import Logger
from loss import SWD, GSWD,H2SWD
from models import  PointNetAE
from models.utils import init_weights
from saver import Saver
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer import AETrainer as Trainer
from utils import get_lr

torch.backends.cudnn.enabled = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config file")
    parser.add_argument("--logdir", help="path to the log directory")
    parser.add_argument("--data_path", help="path to data")
    parser.add_argument("--loss", default="swd",
                        help="[swd, gswd, h2swd]")
    parser.add_argument("--autoencoder", default="pointnet", help="[pointnet]")
    parser.add_argument("--L", default=2, type=int, help="L")
    parser.add_argument("--epoch", default=1000, type=int, help="L")
    parser.add_argument("--r", default=1, type=float, help="T")
    parser.add_argument("--seed", default=1, type=int, help="T")

    args = parser.parse_args()
    config = args.config
    logdir = args.logdir
    data_path = args.data_path
    loss_type = args.loss
    ae_type = args.autoencoder
    r=args.r
    L = args.L
    seed= args.seed
    epoch = args.epoch
    print("Save checkpoints and logs in: ", logdir)
    args = json.load(open(config))
    args["autoencoder"] = ae_type
    args["loss"] = loss_type
    args["num_projs"] = L
    args["num_epochs"] = epoch
    args["seed"]=seed
    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print(">Logdir was created successfully at: ", logdir)

    fname = os.path.join(logdir, "train_ae_config.json")
    with open(fname, "w") as fp:
        json.dump(args, fp, indent=4)

    # print hyperparameters
    print(">You have 5s to check the hyperparameters below.")
    print(args)
    time.sleep(5)

    # init dic of extra parameters for trainer.train
    dic = {}

    # device
    device = torch.device(args["device"])

    # autoencoder architecture
    if args["autoencoder"] == "pointnet":
        autoencoder = PointNetAE(
            args["embedding_size"],
            args["input_channels"],
            args["input_channels"],
            args["num_points"],
            args["normalize"],
        ).to(device)


    else:
        raise Exception("Unknown autoencoder.")

    # loss function
    if args["loss"] == "swd":
        loss_func = SWD(args["num_projs"], device=device)
    elif args["loss"] == "h2swd":
        loss_func = H2SWD(args["num_projs"],r=r,p_type='circular',device= device)
    elif args["loss"] == "gswd":
        loss_func = GSWD(args["num_projs"], r=r,p_type='circular',device=device)
    else:
        raise Exception("Unknown loss function.")

    # dataset
    if args["train_set"] == "shapenet":
        dataset = Shapes3dDataset(dataset_folder=data_path, num_points=args["num_points"], split="train")

    else:
        raise Exception("Unknown dataset")

    # optimizer
    if args["optimizer"] == "sgd":
        optimizer = SGD(
            autoencoder.parameters(),
            lr=args["learning_rate"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"],
        )

    elif args["optimizer"] == "adam":
        optimizer = Adam(
            autoencoder.parameters(),
            lr=args["learning_rate"],
            betas=(0.5, 0.999),
            weight_decay=args["weight_decay"],
        )

    else:
        raise Exception("Optimizer has had implementation yet.")

    # init weights
    if osp.isfile(osp.join(logdir, args["checkpoint"])):
        print(">Init weights with {}".format(args["checkpoint"]))
        checkpoint = torch.load(osp.join(logdir, args["checkpoint"]))
        args["start_epoch"] = checkpoint["epoch"]
        print("Loading from epoch: {}".format(args["start_epoch"]))
        if "autoencoder" in checkpoint.keys():
            autoencoder.load_state_dict(checkpoint["autoencoder"])
        else:
            autoencoder.load_state_dict(checkpoint)
        if "optimizer" in checkpoint.keys():
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                print(">Found no state dict for optimizer.")

    elif osp.isfile(args["checkpoint"]):
        print(">Init weights with {}".format(args["checkpoint"]))
        checkpoint = torch.load(osp.join(args["checkpoint"]))
        if "autoencoder" in checkpoint.keys():
            autoencoder.load_state_dict(checkpoint["autoencoder"])
        else:
            autoencoder.load_state_dict(checkpoint)

    else:
        print(">Init weights with Xavier")
        autoencoder.apply(init_weights)

    # dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        shuffle=True,
        worker_init_fn=seed_worker,
    )

    # logger
    tensorboard_dir = osp.join(logdir, "tensorboard")
    if not osp.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tensorboard_logger = Logger(tensorboard_dir)

    # scheduler
    if args["use_scheduler"]:
        if args["scheduler"] == "cyclic_lr":
            scheduler = CyclicLR(optimizer, base_lr=args["base_lr"], max_lr=args["max_lr"])
        else:
            raise Exception("Unknown learning rate scheduler.")

    # evaluator
    if args["evaluator"] == "based_on_train_loss":
        args["eval_criteria"] = "loss_func"
        args["have_val_set"] = False

    elif args["evaluator"] == "based_on_val_loss":
        args["eval_criteria"] = "loss_func"
        args["have_val_set"] = True

    else:
        raise ValueError("Unknown evaluator.")

    # val_set and val_loader
    if args["have_val_set"]:
        if args["val_set"] == "shapenet":
            val_set = Shapes3dDataset(args["val_root"], num_points=args["num_points"], split="val")

        else:
            raise Exception("Unknown dataset")

        val_loader = DataLoader(
            val_set,
            batch_size=args["val_batch_size"],
            num_workers=args["num_workers"],
            pin_memory=True,
            shuffle=False,
            worker_init_fn=seed_worker,
        )

    # avg_eval_value for model selection
    # init avg_eval_value
    avg_eval_value = args["best_eval_value"]
    best_eval_value = float(args["best_eval_value"])
    best_epoch = int(args["best_epoch"])

    avg_train_loss = args["best_train_loss"]
    best_train_loss = float(args["best_train_loss"])
    best_epoch_based_on_train_loss = int(args["best_epoch_based_on_train_loss"])

    print("best eval value: ", best_eval_value)
    print("best epoch: ", best_epoch)

    # train
    start_epoch = args["start_epoch"]
    num_epochs = args["num_epochs"]

    model_path = os.path.join(logdir, "model.pth")
    best_train_loss_model_path = os.path.join(logdir, "best_train_loss_model.pth")
    f_model_path = os.path.join(logdir, "f_model.pth")
    f_best_train_loss_model_path = os.path.join(logdir, "f_best_train_loss_model.pth")

    rec_train_log_path = os.path.join(logdir, "rec_train.log")
    reg_train_log_path = os.path.join(logdir, "reg_train.log")

    train_log_path = os.path.join(logdir, "train.log")
    eval_log_path = os.path.join(logdir, "eval_when_train.log")

    best_eval_log_path = os.path.join(logdir, "best_eval_when_train.log")
    best_train_log_path = os.path.join(logdir, "best_train.log")

    start_time = time.time()

    dic["iter_id"] = 0
    prev_losses_list = []

    for epoch in tqdm(range(start_epoch, num_epochs)):
        optimizer = SGD(autoencoder.parameters(), lr=0.001)
        dic["curr_epoch"] = epoch

        train_loss_list = []
        rec_train_loss_list = []
        reg_train_loss_list = []

        for batch_id, batch in tqdm(enumerate(train_loader)):
            dic["iter_id"] += 1
            data = batch[0].float().to(device)
            # train_on_batch
            result_dic = Trainer.train(autoencoder, loss_func, optimizer, data, **dic)
            autoencoder = result_dic["ae"]
            optimizer = result_dic["optimizer"]
            train_loss = result_dic["loss"]

            # 2 types of losses
            if "rec_loss" in result_dic.keys():
                rec_train_loss_list.append(result_dic["rec_loss"].item())
            if "reg_loss" in result_dic.keys():
                reg_train_loss_list.append(result_dic["reg_loss"].item())

            # append to loss lists
            train_loss_list.append(train_loss.item())

            # update epsilon for adaptive sw
            if "epsilon" in dic.keys():
                if not args["fix_epsilon"]:
                    # updata prev_losses_list
                    assert ("num_prev_losses" in args.keys()) and (args["num_prev_losses"] > 0)
                    if len(prev_losses_list) == args["num_prev_losses"]:
                        prev_losses_list.pop(0)  # pop the first item
                    prev_losses_list.append(train_loss.item())  # add item to the last
                    dic["epsilon"] = min(prev_losses_list) * args["next_epsilon_ratio_rec"]

            if "rec" in dic.keys() and "epsilon" in dic["rec"].keys():
                dic["rec"]["epsilon"] = result_dic["rec_loss"].item() * args["next_epsilon_ratio_rec"]
            if "reg" in dic.keys() and "epsilon" in dic["reg"].keys():
                dic["reg"]["epsilon"] = result_dic["reg_loss"].item() * args["next_epsilon_ratio_reg"]

            # adjust scheduler
            if args["use_scheduler"]:
                scheduler.step()

            # write tensorboard log
            info = {"train_loss": train_loss.item(), "learning rate": get_lr(optimizer)}
            if "rec_loss" in result_dic.keys():
                info["rec_train_loss"] = rec_train_loss_list[-1]
            if "reg_loss" in result_dic.keys():
                info["reg_train_loss"] = reg_train_loss_list[-1]
            if "num_slices" in result_dic.keys():
                info["num_slices"] = result_dic["num_slices"]
            for tag, value in info.items():
                tensorboard_logger.scalar_summary(tag, value, len(train_loader) * epoch + batch_id + 1)

            # empty cache
            if ("empty_cache_batch" in args.keys()) and args["empty_cache_batch"]:
                torch.cuda.empty_cache()
        # end for 1 epoch

        # calculate avg_train_loss of the epoch
        if len(rec_train_loss_list) > 0:
            avg_rec_train_loss = sum(rec_train_loss_list) / len(rec_train_loss_list)
        if len(reg_train_loss_list) > 0:
            avg_reg_train_loss = sum(reg_train_loss_list) / len(reg_train_loss_list)
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)


        if not args["have_val_set"]:
            avg_eval_value = avg_train_loss

        # save checkpoint
        checkpoint_path = osp.join(logdir, "latest.pth")
        f_checkpoint_path = osp.join(logdir, "f_latest.pth")

        if args["use_scheduler"]:
            Saver.save_checkpoint(autoencoder, optimizer,epoch, checkpoint_path, scheduler=scheduler)
        else:
            Saver.save_checkpoint(autoencoder, optimizer,epoch, checkpoint_path)

        if epoch % args["epoch_gap_for_save"] == 0:
            checkpoint_path = os.path.join(logdir, "epoch_" + str(epoch) + ".pth")
            f_checkpoint_path = os.path.join(logdir, "f_epoch_" + str(epoch) + ".pth")
            Saver.save_best_weights(autoencoder, checkpoint_path)

        # save best model based on avg_eval_value
        if args["eval_criteria"] in ["jsd", "loss_func", "mmd"]:
            better = avg_eval_value < best_eval_value
        elif args["eval_criteria"] in ["cov"]:
            better = avg_eval_value > best_eval_value
        else:
            raise Exception("Unknown eval_criteria")
        if better:
            best_eval_value = avg_eval_value
            best_epoch = epoch
            Saver.save_best_weights(autoencoder, model_path)

        # save best model based on avg_train_loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch_based_on_train_loss = epoch
        if args["evaluator"] != "based_on_train_loss":
            Saver.save_best_weights(autoencoder, best_train_loss_model_path)

        # report
        train_log = "Epoch {}| train_loss : {}\n".format(epoch, avg_train_loss)
        eval_log = "Epoch {}| eval_value : {}\n".format(epoch, avg_eval_value)
        eval_best_log = "Best epoch {}| best eval value: {}\n".format(best_epoch, best_eval_value)
        best_train_loss_log = "Best_train_loss epoch {}| best train loss : {}\n".format(
            best_epoch_based_on_train_loss, best_train_loss
        )
        with open(train_log_path, "a") as fp:
            fp.write(train_log)
        with open(eval_log_path, "a") as fp:
            fp.write(eval_log)
        with open(best_eval_log_path, "w") as fp:
            fp.write(eval_best_log)
        with open(best_train_log_path, "w") as fp:
            fp.write(best_train_loss_log)
        print(train_log)
        print(eval_log)
        print(eval_best_log)
        print(best_train_loss_log)

        if len(rec_train_loss_list) > 0:
            rec_train_log = "Epoch {}| rec_train_loss : {}\n".format(epoch, avg_rec_train_loss)
            with open(rec_train_log_path, "a") as fp:
                fp.write(rec_train_log)
            print(rec_train_log)

        if len(reg_train_loss_list) > 0:
            reg_train_log = "Epoch {}| reg_train_loss : {}\n".format(epoch, avg_reg_train_loss)
            with open(reg_train_log_path, "a") as fp:
                fp.write(reg_train_log)
            print(reg_train_log)

        if ("empty_cache_epoch" in args.keys()) and args["empty_cache_epoch"]:
            torch.cuda.empty_cache()
        print("---------------------------------------------------------------------------------------")
    # end for

    finish_time = time.time()
    total_runtime = finish_time - start_time
    total_runtime = datetime.timedelta(seconds=total_runtime)
    runtime_log = "total runtime: {}".format(str(total_runtime))
    print("total_runtime:", total_runtime)
    with open(train_log_path, "a") as fp:
        fp.write(runtime_log)

    print("Saved checkpoints and logs in: ", logdir)


if __name__ == "__main__":
    main()
