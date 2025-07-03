# export NCCL_DISABLE_P2P=1
result_dir="/playpen/xinyu/crema_result/NeXTQA/"
exp_name='mr_09_pad'
ckpt=${result_dir}${exp_name}/checkpoint_0.pth
ckpt='/playpen/xinyu/crema_result/NeXTQA/nextqa_crema_vtndf_exp3_baseline3/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node=1 --master_port 29203 evaluate.py \
--cfg-path lavis/projects/crema/eval/nextqa_eval.yaml \
--options run.output_dir=${result_dir}test/${exp_name} \
model.finetuned='crema_initial.pth' \
model.mmqa_ckpt=${ckpt} \
model.task='espresso-concat-seq' \
model.frame_num=4 \
model.arch='crema_pad' \
model.missing_mode=3 \
run.batch_size_eval=16 \
model.modalities='rgb_norm_depth_flow' \
datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
model.downstream_task='oeqa'