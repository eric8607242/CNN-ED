import os
import os.path as osp
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import get_model_config, Model
from utils import Criterion, get_logger, AverageMeter, set_random_seed

class Trainer:
    def __init__(self, config):
        # Environment
        # ===================================================================
        self.config = config
        self.device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

        # Dataset
        # ===================================================================
        train_dataset, val_dataset, alphabet_len, max_str_len = \
                get_dataset(path_to_dataset=config["dataset"]["path_to_dataset"],
                            training_set_num=config["dataset"]["training_set_num"], 
                            query_set_num=config["dataset"]["query_set_num"],
                            neighbor_num=config["dataset"]["neighbor_num"])

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=config["dataloader"]["batch_size"],
                                       num_workers=config["dataloader"]["num_workers"],
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=config["dataloader"]["batch_size"],
                                     num_workers=config["dataloader"]["num_workers"],
                                     shuffle=False)

        # Model
        # ===================================================================
        model_config = get_model_config(n_features=config["model"]["n_features"])
        model = Model(model_config, alphabet_len, max_str_len)
        self.model = model.to(self.device)

        # Optimizer
        # ===================================================================
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config["optimizer"]["lr"])

        # Loss Function
        # ===================================================================
        criterion = Criterion(config["criterion"]["alpha"])
        self.criterion = criterion.to(self.device)

        # Training State
        # ===================================================================
        self.current_epoch = -1
        self.current_acc = 0

        # Logger
        # ===================================================================
        get_logger(config["train"]["logdir"])
        self.losses = AverageMeter()
        self.triplet_losses = AverageMeter()
        self.appro_losses = AverageMeter()

    def train(self):
        for epoch in range(self.current_epoch+1, self.config["train"]["n_epochs"]):
            self.current_epoch = epoch
            self._train_one_epoch()
            self._validate()

    def resume(self):
        checkpoint_path = osp.join(self.config["train"]["logdir"], "best.pth")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["current_epoch"]
        self.current_acc = checkpoint["current_acc"]

    def _train_one_epoch(self):
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            anchor_onehot_string, positive_onehot_string, negative_onehot_string, positive_distance, negative_distance = self._data_preprocess(data)
            N = anchor_onehot_string.shape[0]

            anchor_outs = self.model(anchor_onehot_string)
            positive_outs = self.model(positive_onehot_string)
            negative_outs = self.model(negative_onehot_string)

            loss, triplet_loss, appro_loss = self.criterion(anchor_outs, positive_outs, negative_outs, positive_distance, negative_distance)
            loss.backward()

            self.optimizer.step()
            
            self._intermediate_stats_logging(i, len(self.train_loader), loss, triplet_loss, appro_loss, N, "Train")
        self._reset_losses()

    def _intermediate_stats_logging(self, step, len_loader, loss, triplet_loss, appro_loss, N, val_train_state):
        self.losses.update(loss.item(), N)
        self.triplet_losses.update(triplet_loss.item(), N)
        self.appro_losses.update(appro_loss.item(), N)

        if (step > 1 and step % self.config["train"]["print_freq"] == 0) or step == len_loader-1:
            logging.info("{} : [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} Triplet Loss {:.3f} Approx Loss {:.3f}".format(val_train_state, self.current_epoch, self.config["train"]["n_epochs"], step, len_loader-1, self.losses.get_avg(), self.triplet_losses.get_avg(), self.appro_losses.get_avg()))

    def _reset_losses(self):
        self.losses.reset()
        self.triplet_losses.reset()
        self.appro_losses.reset()

    def _validate(self):
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                anchor_onehot_string, positive_onehot_string, negative_onehot_string, positive_distance, negative_distance = self._data_preprocess(data)
                N = anchor_onehot_string.shape[0]

                anchor_outs = self.model(anchor_onehot_string)
                positive_outs = self.model(positive_onehot_string)
                negative_outs = self.model(negative_onehot_string)

                loss, triplet_loss, appro_loss = self.criterion(anchor_outs, positive_outs, negative_outs, positive_distance, negative_distance)
            
                self._intermediate_stats_logging(i, len(self.val_loader), loss, triplet_loss, appro_loss, N, "Val")
            self._reset_losses()
            #self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "current_acc": self.current_acc,
            }
        checkpoint_path = osp.join(self.config["train"]["logdir"], "best.pth")
        torch.save(checkpoint, checkpoint_path)
        logging.info("Save checkpoint to {}".format(checkpoint_path))

    def _data_preprocess(self, data):
        anchor_onehot_string, positive_onehot_string, negative_onehot_string, positive_distance, negative_distance = data
        
        anchor_onehot_string = anchor_onehot_string.to(self.device)
        positive_onehot_string = positive_onehot_string.to(self.device)
        negative_onehot_string = negative_onehot_string.to(self.device)
        
        positive_distance = positive_distance.to(self.device)
        negative_distance = negative_distance.to(self.device)

        return anchor_onehot_string, positive_onehot_string, negative_onehot_string, positive_distance, negative_distance
