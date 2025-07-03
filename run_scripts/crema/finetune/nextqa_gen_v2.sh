result_dir="/playpen/xinyu/crema_result/NeXTQA/"
ckpt='crema_initial.pth'
glw=0.1
for seed in 43 #44
do
exp_name='nextqa_crema_vtn_exp1_ours_fulldata_2_seed'$seed
# exp_name='test'
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run --nproc_per_node=4 --master_port 29003 train.py \
--cfg-path lavis/projects/crema/train/nextqa_gen.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.gen_loss_weight=${glw} \
model.modalities='rgb_norm_depth_flow' \
model.missing_mode=3 \
datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
run.batch_size_train=16 \
run.batch_size_eval=16 \
run.init_lr=1e-4 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.seed=$seed \
run.accum_grad_iters=1 > logs/${exp_name}.log 2>&1
done

# exp_name='nextqa_crema_vtndf_exp3_ours_v2'
# # exp_name='test'
# CUDA_VISIBLE_DEVICES=4,7 python -m torch.distributed.run --nproc_per_node=2 --master_port 29003 train.py \
# --cfg-path lavis/projects/crema/train/nextqa_gen.yaml \
# --options run.output_dir=${result_dir}${exp_name} \
# model.finetuned=${ckpt} \
# model.frame_num=4 \
# model.task='espresso-concat-seq' \
# model.downstream_task='oeqa' \
# model.modalities='rgb_norm_depth_flow' \
# model.missing_mode=3 \
# datasets.nextqa.modality_type=['rgb','norm','depth','flow'] \
# run.batch_size_train=16 \
# run.batch_size_eval=16 \
# run.init_lr=1e-4 \
# run.max_epoch=5 \
# run.warmup_steps=100 \
# run.accum_grad_iters=1 #> logs/${exp_name}.log 2>&1 &