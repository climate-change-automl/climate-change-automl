#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

lr=${lr:-3e-4}
warmup_steps=${warmup_steps:-100}
total_steps=${total_steps:-10000} # Train for 4 epochs
#total_steps=${total_steps:-1} # TODO
layers=${layers:-12}
hidden_size=${hidden_size:-768}
num_head=${num_head:-48}
batch_size=${batch_size:-2}
seed=${seed:-1}
clip_norm=${clip_norm:-5}
blocks=${blocks:-4}
node_loss_weight=${node_loss_weight:-15}
update_freq=${update_freq:-1}

#dt=$(date '+%d%m%Y.%H.%M.%S');
save_dir=./ckpts_automl # TODO remove this every time
tsb_dir=./tsbs # TODO remove this every time
rm -rf $save_dir
rm -rf $tsb_dir
mkdir -p $save_dir
mkdir -p $tsb_dir

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "+++++++++++++++++++++TUNING+++++++++++++++++++++"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "layers: ${layers}"
echo "num_head: ${num_head}"
echo "blocks: ${blocks}"
echo "++++++++++++++++++++++++++++++++++++++++++++++++"
echo "update_freq: ${update_freq}"
echo "total_steps: ${total_steps}"
echo "clip_norm: ${clip_norm}"
echo "blocks: ${blocks}"
echo "node_loss_weight: ${node_loss_weight}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "==============================================================================="

fairseq-train --user-dir ../../graphormer  \
       ./data/is2res_train_val_test_lmdbs/data/is2re/all \
       --valid-subset val_id,val_ood_ads \
       --best-checkpoint-metric loss \
       --num-workers 0 --ddp-backend=c10d \
       --task is2re --criterion mae_deltapos --arch graphormer3d_base  \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm $clip_norm \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $total_steps --batch-size $batch_size \
       --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $tsb_dir \
       --embed-dim $hidden_size --ffn-embed-dim $hidden_size --attention-heads $num_head \
       --max-update $total_steps --log-interval 100 --log-format simple \
       --save-interval-updates 5000 --validate-interval-updates 2500 --keep-interval-updates 30 --no-epoch-checkpoints  \
       --save-dir $save_dir --layers $layers --blocks $blocks --required-batch-size-multiple 1  --node-loss-weight $node_loss_weight #|& tee -a trial.log # TODO get name



# NOTE using the 10k training dataset
