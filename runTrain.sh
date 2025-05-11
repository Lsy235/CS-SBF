#!/bin/bash
# 这里可以放入代码运行命令
echo "program first start..."
echo "please check and update the value of data_path"
echo "model is saved in checkpoints, logger is saved in logs"
python myTrain.py --exp_name "exp_imageGraphFER2013" --data_name "CAER-S" --data_path "databases\\CAER-S" --num_classes 7 --batch_size 16 --num_workers 1 --epoch 40 --R_drop --image_size 224 --pretrainName "swin_base_patch4_window7_224" --pretrainPartPath "swin_base_patch4_window7_224_22kto1k" --k 9 --alphaKLAdj 0.1 --num_node_features 10
