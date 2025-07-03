result_dir="/playpen/xinyu/crema_result/NeXTQA/"
ckpt='crema_initial.pth'

result_dir="/playpen/xinyu/crema_result/NeXTQA/"
ckpt='crema_initial.pth'

for seed in 43 44
do
exp_name='nextqa_crema_vtndf_exp3_baseline2'
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29203 train.py \
--cfg-path lavis/projects/crema/train/nextqa-exp3-bs2.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_norm_depth_flow' \
datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=1e-4 \
run.max_epoch=10 \
run.seed=${seed} \
run.warmup_steps=100 \
run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &
done

# exp_name='nextqa_crema_vtn_exp1_baseline2'
# CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29103 train.py \
# --cfg-path lavis/projects/crema/train/nextqa-exp1-bs2.yaml \
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
# run.warmup_steps=100 \
# run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &

# exp_name='nextqa_crema_vt_allexp_baseline1'
# CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.run --nproc_per_node=2 --master_port 29003 train.py \
# --cfg-path lavis/projects/crema/train/nextqa.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb' \
# datasets.nextqa.modality_type=['rgb'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=10 \
# run.warmup_steps=100 \
# run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &


