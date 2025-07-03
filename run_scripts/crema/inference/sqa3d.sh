result_dir="result/SQA3D/"

# model.finetuned seems no influence
# finetuend model set by: model.mmqa_ckpt='path_to_ft_model'
# model.task='espresso-concat-seq'

exp_name='sqa3d_zs_video+3d'
ckpt='/playpen/xinyu/crema_pretrained.pth'
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nproc_per_node=1 --master_port 29503 evaluate.py \
--cfg-path lavis/projects/crema/eval/sqa3d_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
run.batch_size_eval=16 \
datasets.sqa3d.data_type=['pc','frame','depth','norm'] \
model.downstream_task='oeqa' \
model.modalities='rgb_pc_depth_norm' \
model.mmqa_ckpt='/playpen/xinyu/crema_result/SQA3D/sqa3d_crema_pfdn/checkpoint_best.pth'
