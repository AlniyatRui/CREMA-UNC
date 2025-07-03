result_dir="/playpen/xinyu/crema_result/NeXTQA/"
ckpt='crema_initial.pth'
export NCCL_DISABLE_P2P=1
exp_name='nextqa_crema_vtnd_exp2_ours_v2'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29003 train.py \
--cfg-path lavis/projects/crema/train/nextqa_gen_u4.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_norm_depth' \
model.missing_mode=2 \
datasets.nextqa.modality_type=['rgb','norm','depth'] \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=1e-4 \
run.max_epoch=5 \
run.warmup_steps=100 \
run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &

# 2 modalities exps

# exp_name='nextqa_crema_vd'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-u2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_depth' \
# datasets.nextqa.modality_type=['rgb','depth'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1

# exp_name='nextqa_crema_vn'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-u2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_norm' \
# datasets.nextqa.modality_type=['rgb','norm'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1

# exp_name='nextqa_crema_vf'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-u2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_flow' \
# datasets.nextqa.modality_type=['rgb','flow'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1

# # 3 modalities exps

# exp_name='nextqa_crema_vnd'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-u2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_norm_depth' \
# datasets.nextqa.modality_type=['rgb','norm','depth'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1


# exp_name='nextqa_crema_vnf'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-u2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_norm_flow' \
# datasets.nextqa.modality_type=['rgb','norm','flow'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1

# exp_name='nextqa_crema_vdf'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-u2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_depth_flow' \
# datasets.nextqa.modality_type=['rgb','depth','flow'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1

# all modalities

# exp_name='nextqa_crema_vndf'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29503 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-u2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_depth_norm_flow' \
# datasets.nextqa.modality_type=['rgb','depth','norm','flow'] \
# run.batch_size_train=8 \
# run.batch_size_eval=8 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=2