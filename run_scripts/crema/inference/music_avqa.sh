result_dir="/playpen/xinyu/crema_result/AVQA/"
exp_name='test_avqa_best'
ckpt='/playpen/xinyu/crema_result/AVQA/avqa_crema_vt_allexp_ours/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run --nproc_per_node=4 --master_port 29703 evaluate.py \
--cfg-path lavis/projects/crema/eval/music_avqa_eval.yaml \
--options run.output_dir=${result_dir}/${exp_name} \
model.finetuned='crema_initial.pth' \
model.mmqa_ckpt=${ckpt} \
model.task='espresso-concat-seq' \
model.frame_num=4 \
run.batch_size_eval=16 \
run.seed=42 \
run.find_unused_parameters=True \
datasets.musicavqa_mm_instruct.data_type=['frame','audio','flow','norm','depth'] \
model.modalities='rgb_audio_flow_norm_depth' \
model.downstream_task='oeqa'
