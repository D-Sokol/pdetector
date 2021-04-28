#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import VideoDataset
from models import Net
from preprocess_annotations import grid_centers_1d


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('input')
    parser.add_argument('output', default='result.avi')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--min-proba', '-p', type=float, default=0.5)
    parser.add_argument('--threshold', '-t', type=float, default=0.25)
    return parser


def create_dataloader(args):
    dataset = VideoDataset(args.input, (512,512), args.device)
    return DataLoader(dataset)


def create_model(args):
    model = Net()
    model.load_pretrained(args.weights)
    return model.to(args.device)


def get_boxes(predictions: torch.Tensor, image_shape, p=0.5):
    image_w, image_h = image_shape
    grid_size = predictions.shape[-1]
    assert predictions.shape == (5, grid_size, grid_size)

    cx = torch.as_tensor(grid_centers_1d(image_w, grid_size), dtype=torch.float32)
    cy = torch.as_tensor(grid_centers_1d(image_h, grid_size), dtype=torch.float32)
    cx2d, cy2d = torch.meshgrid(cx, cy)
    cx2d, cy2d = cx2d.T, cy2d.T

    torch.exp_(predictions[3:])
    predictions[1] *= image_w / grid_size
    predictions[2] *= image_h / grid_size
    predictions[3] *= image_w / grid_size
    predictions[4] *= image_h / grid_size

    predictions[1] += cx2d - predictions[3] / 2
    predictions[2] += cy2d - predictions[4] / 2
    predictions[3:5] += predictions[1:3]

    predictions = predictions.view(-1, grid_size**2)
    # [[logit, x1, y1, x2, y2] for _ in boxes if sigmoid(logit) >= p]
    return predictions[:, predictions[0] >= np.log(p / (1-p))]


def _length_1d_intersect(range1, range2):
    return max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))


def boxes_iou(box1, box2):
    #assert len(box1) == len(box2) == 4
    iw = _length_1d_intersect((box1[0], box1[2]), (box2[0], box2[2]))
    ih = _length_1d_intersect((box1[1], box1[3]), (box2[1], box2[3]))
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    intersect = iw * ih
    union = w1 * h1 + w2 * h2 - intersect
    return intersect / union


def non_maximum_suppression(boxes, threshold=0.25):
    N = boxes.shape[1]
    valid = np.ones(N, dtype=bool)
    ixs = np.argsort(boxes[0])
    for i in range(N):
        if not valid[i]:
            continue
        for j in range(i+1, N):
            if valid[j] and boxes_iou(boxes[1:,i], boxes[1:,j]) >= threshold:
                valid[j] = False
    return boxes[1:, valid]


def draw_boxes(frame, boxes):
    for box in boxes.T:
        cv2.rectangle(frame, box[0:2], box[2:4], (255,0,0))


def main(args, writer):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file:///tmp/init', rank=0, world_size=1)

    dl = create_dataloader(args)
    model = create_model(args).to(args.device)

    with torch.no_grad():
        for frame in dl:
            shape = frame.shape[1:3]
            predictions = model(frame).squeeze()
            boxes = get_boxes(predictions, shape, args.min_proba).cpu()
            boxes = non_maximum_suppression(boxes)
            frame = frame.cpu().numpy().transpose(1,2,0)
            draw_boxes(frame, boxes)
            writer.write(frame)


if __name__ == '__main__':
    args = create_parser().parse_args()
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'DIVX'), 15, (512, 512))
    main(args, writer)
    writer.release()

