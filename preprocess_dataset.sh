#!/bin/bash
mkdir -p checkpoints/day2nightA
cp extra/Day2Night_model/checkpoints/daynight_cyclegan_new/latest_net_D_A.pth checkpoints/day2nightA/latest_net_D.pth
cp extra/Day2Night_model/checkpoints/daynight_cyclegan_new/latest_net_G_A.pth checkpoints/day2nightA/latest_net_G.pth

mkdir dataroot -p
unzip extra/dhd.zip -d dataroot
rm dataroot/dhd_traffic/images/train/1502433988638.jpg
rm dataroot/dhd_traffic/images/train/1502445483179.jpg

for subset in 'train' 'val'; do
    rm -rf results/

    python3 pytorch-CycleGAN-and-pix2pix/test.py --eval \
        --phase test \
        --name day2nightA \
        --dataroot dataroot/dhd_traffic/images/${subset} \
        --model test \
        --num_test 100000 \
        --serial_batches \
        --no_dropout \
        --batch_size 256

    for file in results/day2nightA/test_latest/images/*real*; do
        rm $file
    done
    for file in results/day2nightA/test_latest/images/*; do
        mv $file ${file/_fake/}
    done

    zip processed_${subset}.zip -r results/day2nightA/test_latest/images
    cp processed_${subset}.zip extra/processed_${subset}.zip
done

