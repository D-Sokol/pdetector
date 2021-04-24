#!/usr/bin/env python3
from collections import defaultdict
import json
import numpy as np
import pickle
import tqdm


def grid_centers_1d(image_size, grid_size):
    arr, step = np.linspace(0, image_size, grid_size, endpoint=False, retstep=True)
    arr += step / 2
    return arr


def annot2targets(annotations: dict, image_shapes: dict, grid_size: int):
    targets_all = defaultdict(lambda : np.zeros((1+4, grid_size, grid_size)))

    prev_shape = None
    for annot in tqdm.tqdm(annotations['annotations']):
        x1, y1, w, h = annot['bbox']
        targets = targets_all[annot['image_id']]
        image_w, image_h, _unused = image_shapes[annot['image_id']]
        if prev_shape != (image_w, image_h):
            cx = grid_centers_1d(image_w, grid_size)
            cy = grid_centers_1d(image_h, grid_size)
            cx2d, cy2d = np.meshgrid(cx, cy)
            prev_shape = (image_w, image_h)

        box_mask_x = (x1 <= cx) & (x1 + w > cx)
        if not box_mask_x.any():
            box_mask_x[np.abs(x1 + w/2 - cx).argmin()] = True
        box_mask_y = (y1 <= cy) & (y1 + h > cy)
        if not box_mask_y.any():
            box_mask_y[np.abs(y1 + h/2 - cy).argmin()] = True
        box_mask = box_mask_x.reshape(1, -1) & box_mask_y.reshape(-1, 1)

        targets[0, box_mask] = 1
        targets[1, box_mask] = (x1 + w/2 - cx2d[box_mask]) / image_w * grid_size
        targets[2, box_mask] = (y1 + h/2 - cy2d[box_mask]) / image_h * grid_size
        targets[3, box_mask] = np.log(w / image_w * grid_size)
        targets[4, box_mask] = np.log(h / image_h * grid_size)
    return dict(targets_all)



if __name__ == '__main__':
    grid_size = 15

    for subset in ('train', 'val'):
        with open(f'extra/annotations/dhd_pedestrian_traffic_{subset}.json') as file:
            annotations = json.load(file)

        image_shapes = {
            x['id']: (x['width'], x['height'], x['file_name'])
            for x in annotations['images']
        }

        targets = annot2targets(annotations, image_shapes, grid_size)
        with open(f'extra/annotations/annotations_{subset}.pkl', 'wb') as file:
            pickle.dump(targets, file)

