mkdir -p log_PACKD_imagenet
python train_student_imagenet.py \
 --dist-url 'tcp://127.0.0.1:55515' \
 --root_dir ./data/images/ \
 --dist-backend 'nccl' \
 --multiprocessing-distributed \
 --checkpoint-dir ./checkpoint/ \
 --batch-size 1024 \
 --num-workers 16 \
 --gpu 0,1,2,3,4,5,6,7 \
 --tcheckpoint ./pretrained_models/resnet34-333f7ec4.pth \
 --world-size 1 --rank 0 --manual_seed 0 
