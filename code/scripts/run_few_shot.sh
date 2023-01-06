model_teachers=( resnet56 )
model_students=( resnet20 )


run_exp(){
model_teacher=$1
model_student=$2
kd_methods=(PACKDCONLoss PACKDCONLoss PACKDCONLoss)
    alphas=(   1    1   1   )
alpha_augs=(   1    1   1   )
     betas=(   0.8  0.8 0.8 )
   few_ratios=(0.25 0.5 0.70)
   mixup_nums=(3    3   3   )
       trials=(1    2   3   )

for(( i=0;i<${#kd_methods[@]};i++)) do
    currenttime=`date "+%Y%m%d_%H%M%S"`
    work_path=./log_PACKD_fewshot_mixnum/${model_teacher}_${model_student}/
    if [ ! -d $work_path ]; then
    mkdir -p $work_path
    fi
    jobname=$(pwd | awk -F "/" '{print $NF}')
    gpu_num=1
    kd_method=${kd_methods[i]}
    alpha=${alphas[i]}
    alpha_aug=${alpha_augs[i]}
    beta=${betas[i]}
    mixup_num=${mixup_nums[i]}
    few_ratio=${few_ratios[i]}
    trial=${trials[i]}
    echo ${kd_method};
    python train_student.py --dataset cifar100 \
    --path_t ./save_t/models/${model_teacher}_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth  \
    --distill ${kd_method} --model_s ${model_student} \
    --save_dir save_PACKD/ -a ${alpha} --alpha_aug  ${alpha_aug} -b ${beta} --mixup_num ${mixup_num} \
    --few-ratio ${few_ratio} \
    --trial ${i} 2>&1 | tee ${work_path}/${kd_method}_alpha_${alpha}_alphaaug_${alpha_aug}_beta_${beta}_mixup_num_${mixup_num}_trial_${trial}_${currenttime}.log 
    sleep 3s
done;
}

for(( j=0;j<${#model_teachers[@]};j++)) do
    echo ${model_teachers[j]} ${model_students[j]} $j
    run_exp ${model_teachers[j]} ${model_students[j]}
    #sleep 60m;
done;

