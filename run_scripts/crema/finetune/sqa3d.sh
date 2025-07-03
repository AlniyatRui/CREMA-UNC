result_dir="/playpen/xinyu/crema_result/SQA3D/"

exp_name='sqa3d_crema_vt_allexp_baseline1'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.run --nproc_per_node=2 --master_port 29603 train.py \
--cfg-path lavis/projects/crema/train/sqa3d.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb' \
datasets.sqa3d.data_type=['frame'] \
run.batch_size_train=16 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &

# exp_name='sqa3d_crema_vtnpf_exp3_baseline2'
# ckpt='crema_initial.pth'
# CUDA_VISIBLE_DEVICES=4,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29103 train.py \
# --cfg-path lavis/projects/crema/train/sqa3d-exp3-bs2.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_pc_depth_norm' \
# datasets.sqa3d.data_type=['pc','frame','depth','norm'] \
# run.batch_size_train=16 \
# run.batch_size_eval=24 \
# run.init_lr=2e-4 \
# run.max_epoch=20 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &