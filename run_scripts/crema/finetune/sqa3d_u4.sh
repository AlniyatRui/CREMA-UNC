result_dir="/playpen/xinyu/crema_result/SQA3D/"
# export NCCL_DISABLE_P2P=1
exp_name='sqa3d_crema_pfdn'
ckpt='crema_initial.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 29603 train.py \
--cfg-path lavis/projects/crema/train/sqa3d-u4.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_pc_depth_norm' \
datasets.sqa3d.data_type=['pc','frame','depth','norm'] \
run.batch_size_train=16 \
run.batch_size_eval=24 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &
