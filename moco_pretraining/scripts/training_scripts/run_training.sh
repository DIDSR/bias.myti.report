#!/bin/bash
# # script to run the training
# #
# # architectures: choose from 'alexnet', 'densenet121', 'densenet161', 'densenet169', 
# #     'densenet201', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 
# #     'mnasnet1_3', 'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 
# #     'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 
# #     'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 
# #     'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2'
# # Variable descriptions
# #  --from-imagenet: use pre-trained ImageNet model
# #  --moco-t: softmax temperature (default: 0.07)
# #  --world-size: number of nodes for distributed training
# #
# # resnet18: --lr 0.0001
# # densenet121:  --lr 0.0001
# # wide_resnet50_2:   --lr 0.0001
cd ../../moco; python main_moco.py -a wide_resnet50_2 \
            --lr 0.0001 --batch-size 16 \
            --epochs 20 \
            --world-size 1 --rank 0 \
            --mlp --moco-t 0.2 --from-imagenet \
            --dist-url 'tcp://127.0.0.1:10001' --multiprocessing-distributed \
			--aug-setting chexpert --rotate 10 --maintain-ratio \
            --train_data /gpfs_projects/ravi.samala/DATA/CheXpert-v1.0-small/train/ \
            --exp-name wide_resnet50_2_w00001_20220819h09_tr

# done
echo "Done"
