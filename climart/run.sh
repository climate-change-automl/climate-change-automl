#!/bin/bash

python main.py --model "GCN+Readout" --exp_type "pristine" --target_type "shortwave" --workers 1 --seed 7 \
	  --batch_size 128 --lr 2e-4 --optim Adam --weight_decay 1e-6 --scheduler "expdecay" --in_normalize "Z" --net_norm "layer_norm" --dropout 0.0 --act "GELU" --epochs 100 \
	      --preprocessing "mlp_projection" --projector_net_normalization "layer_norm" --graph_pooling "mean" --residual --improved_self_loops \
		  --gradient_clipping "norm" --clip 1.0 --hidden_dims 128 128 128 ---train_years "1990+1999+2003" --validation_years "2005" \
		      --wandb_mode online --load_train_into_mem --load_val_into_mem --device cuda \
