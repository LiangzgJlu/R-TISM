import os
import torch
import datetime
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from dataset.driver_feature_dataset import DriverModelDataset
from model.driver_feature_model import DriverFeatureModel
from loguru import logger


def train(config: dict):

    dataset = DriverModelDataset(
        config["track_path_15"],
        config["track_path_30"],
        config["history_window_length"],
        config["noise"])

    data_loader = DataLoader(dataset=dataset, batch_size=config["batch_size"])

    model = DriverFeatureModel(4, 200, 2)
    model.to(config["device"])

    size = dataset.__len__()
    # 评价标准
    criterion = nn.MSELoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # 学习率方案
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # 每30个epoch学习率乘以0.1  
    min_loss = 100000
    losses = []
    for i_episode in range(int(config["epochs"])):
        total_loss = float(0)
        for inputs, targets in data_loader:
            inputs = inputs.type(dtype=torch.float32).to(config["device"])
            targets = targets.type(dtype=torch.float32).to(config["device"])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # exit()        
        scheduler.step()
        losses.append(total_loss)

        logger.info("epoch: {}, total loss: {}, mean loss: {}.".format(i_episode, total_loss, total_loss / size))
        if (min_loss > total_loss):
            min_loss = total_loss
            torch.save(model.state_dict(),
                       config["checkpoint_save_path"])
            logger.info("save model to {}".format(config["checkpoint_save_path"]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='DriverModelTrainer')
    parser.add_argument('-lp', '--log_path', default="/home/public/lzg/car-following-model/log", type=str)
    parser.add_argument('-hwl', '--history_windows_length', default=25, type=int)  # option that takes a value
    parser.add_argument('-bs', '--batch_size', default=512, type=int)
    parser.add_argument('-e', '--epochs', default=400, type=int)
    parser.add_argument('-tp15', '--track_path_15', type=str)
    parser.add_argument('-tp30', '--track_path_30', type=str)
    parser.add_argument('-lc', '--load_checkpoint', type=str, default=None)
    parser.add_argument('-csp', '--checkpoint_save_path', type=str, default=None)
    parser.add_argument('-n', '--noise', type=float, default=0.0)
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()

    # configure logger
    run_name = os.path.basename(__file__).split(".")[0]
    log_path = args.log_path + os.sep + run_name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.add(log_path)
    logger.info(args.__str__())

    if args.train:
        train_config = {
            "history_window_length": args.history_windows_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "track_path_15": args.track_path_15,
            "track_path_30": args.track_path_15,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "load_checkpoint": args.load_checkpoint,
            "checkpoint_save_path": args.checkpoint_save_path,
            "noise": args.noise
        }
        train(train_config)
