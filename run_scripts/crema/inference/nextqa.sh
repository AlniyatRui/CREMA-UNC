# export NCCL_DISABLE_P2P=1
result_dir="/playpen/xinyu/crema_result/NeXTQA/"
exp_name='mr_09_gen'
ckpt=${result_dir}${exp_name}/checkpoint_0.pth
ckpt='/home/xinyuzh/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port 29103 evaluate.py \
--cfg-path lavis/projects/crema/eval/nextqa_eval.yaml \
--options run.output_dir=${result_dir}test/${exp_name} \
model.finetuned='crema_initial.pth' \
model.mmqa_ckpt=${ckpt} \
model.task='espresso-concat-seq' \
model.frame_num=4 \
model.missing_mode=3 \
run.batch_size_eval=16 \
model.modalities='rgb_norm_depth_flow' \
datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
model.downstream_task='oeqa'