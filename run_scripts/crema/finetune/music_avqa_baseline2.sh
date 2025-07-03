result_dir="/playpen/xinyu/crema_result/AVQA/"

exp_name='avqa_crema_vtandf_baseline2'
ckpt='crema_initial.pth'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node=1 --master_port 29513 train.py \
--cfg-path lavis/projects/crema/train/musicavqa-bs2.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
datasets.musicavqa_mm_instruct.data_type=['audio','frame','flow','norm','depth'] \
model.modalities='rgb_audio_flow_norm_depth' \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 #> logs/${exp_name}.log 2>&1 &
