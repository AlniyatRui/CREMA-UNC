result_dir="/playpen/xinyu/crema_result/NeXTQA/"
ckpt='crema_initial.pth'

# exp_name='nextqa_crema_vtndf_exp3_ours_v2'
exp_name='test'
CUDA_VISIBLE_DEVICES=0,1,2,4 python -m torch.distributed.run --nproc_per_node=2 --master_port 29903 train.py \
--cfg-path lavis/projects/crema/train/nextqa_gen.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.missing_mode=1 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_norm' \
datasets.nextqa.modality_type=['rgb','norm'] \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=1e-4 \
run.max_epoch=100 \
run.warmup_steps=100 \
run.accum_grad_iters=1 #> logs/${exp_name}.log 2>&1 &
# model.missing_mode=2 \
