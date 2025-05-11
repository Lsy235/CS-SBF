# Paper is Published in IJCAI2025: Fusion of Granular-Ball Visual Spatial Representations for Enhanced Facial Expression Recognition

## main environment:
	numpy                     1.23.5
	opencv-python          4.6.0.66
	pandas                     2.0.3
	timm                        1.0.11
	torch                        2.4.1+cu124
	torch-geometric         2.6.1
	torch_scatter             2.1.2+pt24cu124
	torch_sparse             0.6.18+pt24cu124
	tqdm                        4.66.5
	torchvision                0.19.1+cu124
	
## run:
	Step 1. Download the CAER-S database into the databases folder.
	
	Step 2. Retrieve the patch4_window7_224_22kto1k.pth from https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth to pretrainModel folder.
	
	Step 3. run utils/imgToGraph.py to get the graph data represented by the multi-granularity balls.
	
	Step 4. run the runTrain.sh script to train and evaluate the model.
	
