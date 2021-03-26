import os
import os.path as osp
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import get_dataset
from model import get_model_config, Model
from utils import Criterion, get_logger, AverageMeter, set_random_seed, evaluate, get_writer

class Trainer:
    def __init__(self, config):
        # Environment
        # ===================================================================
        self.config = config
        self.device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

        # Dataset
        # ===================================================================
        train_dataset, query_dataset, base_dataset, alphabet_len, max_str_len = \
                get_dataset(path_to_dataset=config["dataset"]["path_to_dataset"],
                            training_set_num=config["dataset"]["training_set_num"], 
                            query_set_num=config["dataset"]["query_set_num"],
                            neighbor_num=config["dataset"]["neighbor_num"])

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=config["dataloader"]["batch_size"],
                                       num_workers=config["dataloader"]["num_workers"],
                                       shuffle=True)
        self.query_loader = DataLoader(dataset=query_dataset,
                                       batch_size=config["dataloader"]["batch_size"],
                                       num_workers=config["dataloader"]["num_workers"],
                                       shuffle=False)
        self.base_loader = DataLoader(dataset=base_dataset,
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
        self.writer = get_writer(config["train"]["logdir_tb"])
        get_logger(config["train"]["logdir"])
        self.losses = AverageMeter()
        self.triplet_losses = AverageMeter()
        self.appro_losses = AverageMeter()

    def train(self):
        best_recall = 0
        for epoch in range(self.current_epoch+1, self.config["train"]["n_epochs"]):
            self.current_epoch = epoch
            self._train_one_epoch()
            recall = self._validate()

            if recall > best_recall:
                self._save_checkpoint()
                best_recall = recall

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

        self.writer.add_scalar("Train_Loss/total_loss", self.losses.get_avg(), self.current_epoch)
        self.writer.add_scalar("Train_Loss/triplet_loss", self.triplet_losses.get_avg(), self.current_epoch)
        self.writer.add_scalar("Train_Loss/approximation_loss", self.appro_losses.get_avg(), self.current_epoch)
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
            query_outs_list = []
            query_distance = []
            for i, data in enumerate(self.query_loader):
                anchor_onehot_string, anchor_distance = data
                anchor_onehot_string = anchor_onehot_string.to(self.device)
                anchor_distance = anchor_distance.to(self.device)

                anchor_outs = self.model(anchor_onehot_string)

                query_outs_list.append(anchor_outs)
                query_distance.append(anchor_distance)

            query_outs = torch.cat(query_outs_list)
            query_distance = torch.cat(query_distance)

            base_outs_list = []
            for i, data in enumerate(self.base_loader):
                anchor_onehot_string = data
                anchor_onehot_string = anchor_onehot_string.to(self.device)

                anchor_outs = self.model(anchor_onehot_string)

                base_outs_list.append(anchor_outs)
            base_outs = torch.cat(base_outs_list)

            average_recall = evaluate(query_outs, base_outs, query_distance, self.config["evaluate"]["K"])
            logging.info("Val : [{:3d}/{}] Evaluate recall (K : {}) : {:.4f}".format(self.current_epoch, self.config["train"]["n_epochs"], self.config["evaluate"]["K"], average_recall))

        self.writer.add_scalar("Val_Recall/recall", average_recall, self.current_epoch)
        return average_recall


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
        """
            Unpack data and convert to the training environment device
        """
        anchor_onehot_string, positive_onehot_string, negative_onehot_string, positive_distance, negative_distance = data
        
        anchor_onehot_string = anchor_onehot_string.to(self.device)
        positive_onehot_string = positive_onehot_string.to(self.device)
        negative_onehot_string = negative_onehot_string.to(self.device)
        
        positive_distance = positive_distance.to(self.device)
        negative_distance = negative_distance.to(self.device)

        return anchor_onehot_string, positive_onehot_string, negative_onehot_string, positive_distance, negative_distance
