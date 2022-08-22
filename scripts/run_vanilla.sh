# sample scripts for training vanilla teacher models
#spring.submit run --gpu \
#"python train_teacher.py --model wrn_40_2"
#exit
currenttime=`date "+%Y%m%d_%H%M%S"`
work_path=./log/
jobname=$(pwd | awk -F "/" '{print $NF}')
gpu_num=1


python train_teacher.py --model wrn_40_2 2>&1 |tee ${work_path}/wrn_40_2_${currenttime}.log
sleep 1s
python train_teacher.py --model resnet56 2>&1 |tee ${work_path}/resnet56_${currenttime}.log
sleep 1s
python train_teacher.py --model resnet110 2>&1 |tee ${work_path}/resnet110_${currenttime}.log
sleep 1s
python train_teacher.py --model resnet32x4 2>&1 |tee ${work_path}/resnet32x4_${currenttime}.log
sleep 1s
python train_teacher.py --model vgg13 2>&1 |tee ${work_path}/vgg13_${currenttime}.log
sleep 1s
python train_teacher.py --model ResNet50 2>&1 |tee ${work_path}/ResNet50_${currenttime}.log
