#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import PDDataset
from models import Net
from loss import YOLOLoss


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--annotations-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val-size', type=float, default=0.2)

    parser.add_argument('--spatial-coef', type=float, default=5.0)
    parser.add_argument('--positive-coef', type=float, default=2.0)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-ddr', type=float, default=0.0001)
    parser.add_argument('--lr-steps', type=int, default=20)
    parser.add_argument('--lr-gamma', type=float, default=0.1)

    parser.add_argument('--checkpoints-dir', default='checkpoints')
    parser.add_argument('--ddr-weights', default='extra/DDRNet23s.pth')
    parser.add_argument('--checkpoints-freq', type=int, default=5)
    parser.add_argument('--start-epoch', type=int, default=0)

    parser.add_argument('--verbose', action='store_true')
    return parser


def create_dataloaders(args):
    with open(args.annotations_path, 'rb') as file:
        targets = pickle.load(file)
    dataset = PDDataset((args.data_path,), targets, device=args.device)

    train_size = round(len(dataset) * args.val_size)
    test_size = len(dataset) - train_size
    data_train, data_test = random_split(dataset, (train_size, test_size))

    dl_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)
    return dl_train, dl_test


def create_model(args):
    model = Net()
    if args.start_epoch:
        path = os.path.join(args.checkpoints_dir, f'checkpoint-{args.start_epoch}.pth')
        model.load_state_dict(torch.load(path))
    else:
        model.load_pretrained(args.ddr_weights)
    return model.to(args.device)


def create_optimizer(args, model: Net):
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR

    opt1 = Adam(model.partial_parameters(True), lr=args.lr_ddr)
    opt2 = Adam(model.partial_parameters(False), lr=args.lr)

    lrs1 = StepLR(opt1, args.lr_steps, args.lr_gamma)
    lrs1.last_epoch = args.start_epoch - 1
    lrs2 = StepLR(opt2, args.lr_steps, args.lr_gamma)
    lrs2.last_epoch = args.start_epoch - 1
    return opt1, opt2, lrs1, lrs2


def main(args):
    dl_train, dl_test = create_dataloaders(args)
    if args.verbose:
        dl_train, dl_test = tqdm(dl_train), tqdm(dl_test)

    model = create_model(args).to(args.device)
    opt1, opt2, lrs1, lrs2 = create_optimizer(args, model)
    loss_layer = YOLOLoss(args.spatial_coef, args.positive_coef)

    for epoch in range(args.start_epoch, args.epochs):
        losses = []

        model.train(True)
        for input, targets in dl_train:
            output = model(input)
            loss = loss_layer(output, targets)
            loss.backward()
            opt1.step()
            opt2.step()
            lrs1.step()
            lrs2.step()
            opt1.zero_grad()
            opt2.zero_grad()
            losses.append(loss.item())

        loss_train = np.mean(losses)
        losses.clear()

        model.train(False)
        with torch.no_grad():
            for input, targets in dl_test:
                output = model(input)
                loss = loss_layer(output, targets)
                losses.append(loss.item())

        loss_test = np.mean(losses)

        print("Epoch {:2d}/{:2d}\tTrain loss {:6.2f}\tTest loss {:6.2f}".format(
            epoch, args.epochs,
            loss_train,
            loss_test
        ))

        if epoch and epoch % args.checkpoints_freq == 0:
            path = os.path.join(args.checkpoints_dir, f'checkpoint-{epoch}.pth')
            with open(path, 'wb') as file:
                torch.save(model.state_dict(), file)


if __name__ == '__main__':
    args = create_parser().parse_args()
    main(args)

