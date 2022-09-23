#!/bin/bash

clip_norm=0.0
warmup_updates=0

fairseq-train --user-dir ../../graphormer  \
	   ./data/is2res_train_val_test_lmdbs/data/is2re/all --valid-subset val_id,val_ood_ads,val_ood_cat,val_ood_both --best-checkpoint-metric loss \
	      --num-workers 0 --ddp-backend=c10d \
	         --task is2re --criterion mae_deltapos --arch graphormer3d_base  \
		    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm $clip_norm \
		       --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates $warmup_updates --total-num-update 1000000 --batch-size 4 \
		          --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 1 --seed 1 \
			     --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir ./tsbs \
			        --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 \
				   --max-update 1000000 --log-interval 100 --log-format simple \
				      --save-interval-updates 5000 --validate-interval-updates 2500 --keep-interval-updates 30 --no-epoch-checkpoints  \
				         --save-dir ./ckpt --layers 12 --blocks 4 --required-batch-size-multiple 1  --node-loss-weight 15
