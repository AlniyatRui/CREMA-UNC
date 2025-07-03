export LD_LIBRARY_PATH=/data/workspace/zhangjunrui/.conda/envs/crema/lib:$LD_LIBRARY_PATH
LD_PRELOAD=/data/workspace/zhangjunrui/.conda/envs/crema/lib/libstdc++.so.6
result_dir="crema_result/SQA3D/"

# 1. 改攻击方法与训练框架
# 2. 改AT_EXP作为同框架实验对比
# 3. 修改exp_name

export AT=True
export ATTACK_METHOD="NuAT" # optional: FGSM(for FGSM-PCO), MI(available, for TRADES), NuAT
export AT_FRAMEWORK="NuAT" # optional: TRADES(avalible), NuAT, FGSM-PCO
export AT_EXP="" #optional: ""(Ours) "Avg" "Random" "Rgb"(Only Atk Video)

export ATTACK_RATIO=0.1 # ratio of at samples
export ATTACK_ITERS=5 # only for MI(Multi-steps) iterations
export ATTACK_EPS=8 # attack eps in training
export SOFTMAX_TEMP=0.2 # softmax temperature in adaptive weight
export PREATTACK_ALPHA=2 # pre-attack alpha in one-step loss ascend phase
export ALPHA=2
export MODALITIES='rgb_depth_norm'

exp_name='10%-NuAT-PreAttack'
ckpt='/data/workspace/zhangjunrui/NeurIPS25/CREMA/SQA3D/sqa3d_vpdn.pth'
# resume_ckpt_path='/data/workspace/zhangjunrui/NeurIPS25/CREMA/SQA3D/sqa3d_vpdn.pth'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=1 --master_port 29413 train.py \
--cfg-path lavis/projects/crema/train/sqa3d-exp3-bs2.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
model.frame_num=4 \
model.task='espresso-concat-seq' \
model.downstream_task='oeqa' \
model.modalities='rgb_pc_depth_norm' \
datasets.sqa3d.data_type=['pc','frame','depth','norm'] \
run.batch_size_train=1 \
run.batch_size_eval=16 \
run.init_lr=2e-4 \
run.max_epoch=5 \
run.warmup_steps=1000 \
run.accum_grad_iters=16 #> logs/${exp_name}.log 2>&1 &
# run.resume_ckpt_path=${resume_ckpt_path} \