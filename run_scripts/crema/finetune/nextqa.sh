result_dir="/playpen/xinyu/crema_result/NeXTQA/"
ckpt='crema_initial.pth'

exp_name='nextqa_crema_vtn_exp1_baseline3'
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29203 train.py \
--cfg-path lavis/projects/crema/train/nextqa-pad.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_norm' \
datasets.nextqa.modality_type=['rgb','norm'] \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=1e-4 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 #> logs/${exp_name}.log 2>&1

# exp_name='nextqa_crema_vtnd_exp2_baseline3'
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29203 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-pad.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_norm_depth' \
# datasets.nextqa.modality_type=['rgb','norm','depth'] \
# run.batch_size_train=8 \
# run.batch_size_eval=8 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=2 > logs/${exp_name}.log 2>&1 &

# exp_name='nextqa_crema_vtndf_exp3_baseline3'
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29203 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-pad.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_norm_depth_flow' \
# datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
# run.batch_size_train=8 \
# run.batch_size_eval=8 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=1000 \
# run.accum_grad_iters=2 > logs/${exp_name}.log 2>&1 &