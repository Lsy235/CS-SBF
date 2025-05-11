main environment:
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
run:
	1. Download the CAER-S database into the databases folder.
	2. Retrieve the patch4_window7_224_22kto1k.pth from https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth to pretrainModel folder.
	3. run utils/imgToGraph.py to get the graph data represented by the multi-granularity balls.
	4. run the runTrain.sh script to train and evaluate the model.
	
  In this supplement, we provide additional ablation experiments as well as experimental source code, experimental source data, and we have run through this code on new machines. Due to size limitations of the supplemental material submission and material preparation time limitations, the databases is not available in the Supplementary Material and needs to be downloaded on your own. Due to the size limitation of the supplementary material, we could not provide the intermediate data of the granular balls representation and the trained model weight file, and could only obtain them by running the code on our own through steps 3 and 4. If the subsequent paper is accepted, we will publicize all the experimental data as well as the trained model on the research team's github.
