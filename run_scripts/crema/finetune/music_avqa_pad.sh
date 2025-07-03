result_dir="/playpen/xinyu/crema_result/AVQA/"

exp_name='avqa_crema_vt_allexp_ours'
ckpt='crema_initial.pth'

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run --nproc_per_node=4 --master_port 29903 train.py \
--cfg-path lavis/projects/crema/train/music_avqa_gen.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.missing_mode=4 \
datasets.musicavqa_mm_instruct.data_type=['frame','audio','flow','norm','depth'] \
model.modalities='rgb_audio_flow_norm_depth' \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=2e-4 \
run.max_epoch=20 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1 &
