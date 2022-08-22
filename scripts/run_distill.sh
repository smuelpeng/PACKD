model_teachers=( wrn_40_2 wrn_40_2 resnet56  resnet32x4 vgg13 vgg13       ResNet50    ResNet50 resnet32x4 resnet32x4 wrn_40_2 )
model_students=( wrn_16_2 wrn_40_1 resnet20  resnet8x4  vgg8  MobileNetV2 MobileNetV2 vgg8     ShuffleV1 ShuffleV2   ShuffleV1)

run_exp(){
model_teacher=$1
model_student=$2
kd_methods=(PACKDCONLoss)
    gammas=(1)
    alphas=(1)
alpha_augs=(1)
     betas=(0.8)
mixup_nums=(3)
    trials=(1)

for(( i=0;i<${#kd_methods[@]};i++)) do
    currenttime=`date "+%Y%m%d_%H%M%S"`
    work_path=./log_PACKD/${model_teacher}_${model_student}/
    if [ ! -d $work_path ]; then
    mkdir -p $work_path
    fi
    jobname=$(pwd | awk -F "/" '{print $NF}')
    gpu_num=1
    kd_method=${kd_methods[i]}
    alpha=${alphas[i]}
    alpha_aug=${alpha_augs[i]}
    beta=${betas[i]}
    gamma=${gammas[i]}
    mixup_num=${mixup_nums[i]}
    trial=${trials[i]}
    echo ${kd_method};
    python train_student.py --dataset cifar100 \
    --path_t ./save_t/${model_teacher}_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth  \
    --distill ${kd_method} --model_s ${model_student} --pos_k ${pos_k} \
    --save_dir save_PACKD/ -r ${gamma} -a ${alpha} --alpha_aug  ${alpha_aug} -b ${beta} --mixup_num ${mixup_num} \
    --trial ${i} 2>&1 |tee ${work_path}/${kd_method}_gamma_${gamma}_alpha_${alpha}_alphaaug_${alpha_aug}_beta_${beta}_mixup_num_${mixup_num}_trial_${trial}_${currenttime}.log
    sleep 3s
done;
}

for(( j=0;j<${#model_teachers[@]};j++)) do
    echo ${model_teachers[j]} ${model_students[j]} $j
    run_exp ${model_teachers[j]} ${model_students[j]}
done;
