#!/bin/bash
# # script to run the training
cd ../../moco; python main_moco.py -a resnet18 \
            --lr 0.0001 --batch-size 16 \
            --epochs 20 \
            --world-size 1 --rank 0 \
            --mlp --moco-t 0.2 --from-imagenet \
            --dist-url 'tcp://127.0.0.1:10001' --multiprocessing-distributed \
			--aug-setting chexpert --rotate 10 --maintain-ratio \
            --train_data /gpfs_projects/ravi.samala/DATA/CheXpert-v1.0-small/train/ \
            --exp-name r8w1n416_20220715h15_tr

# done
echo "Done"
