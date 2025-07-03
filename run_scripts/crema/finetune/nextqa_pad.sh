result_dir="/playpen/xinyu/crema_result/NeXTQA/"
ckpt='crema_initial.pth'
resume_ckpt_path='/playpen/xinyu/crema_result/NeXTQA/nextqa_crema_vtndf_exp3_baseline3/checkpoint_best.pth'
# exp_name='nextqa_crema_vtn_exp1_baseline3'
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node=4 --master_port 29103 train.py \
# --cfg-path lavis/projects/crema/train/nextqa_pad.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.missing_mode=1 \
# model.modalities='rgb_norm' \
# datasets.nextqa.modality_type=['rgb','norm'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1

# exp_name='nextqa_crema_vtnd_exp2_baseline3'
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port 29803 train.py \
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node=4 --master_port 29803 train.py \
# --cfg-path lavis/projects/crema/train/nextqa_pad.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.missing_mode=2 \
# model.modalities='rgb_norm_depth' \
# datasets.nextqa.modality_type=['rgb','norm','depth'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 
for seed in 43 44
do
exp_name='nextqa_crema_vtndf_exp3_baseline3_'$seed
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port 29903 train.py \
--cfg-path lavis/projects/crema/train/nextqa_pad.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.missing_mode=3 \
model.modalities='rgb_norm_depth_flow' \
datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=1e-4 \
run.max_epoch=5 \
run.warmup_steps=100 \
run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1
done