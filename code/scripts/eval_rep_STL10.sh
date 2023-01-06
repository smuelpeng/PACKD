mkdir -p log_rep
python eval_rep_base.py  --data data/ \
     --dataset STL-10 \
     --arch resnet8x4 \
     --s-path save_PACKD/student_PACKDCONLoss_model/S:resnet8x4_T:resnet32x4_cifar100_PACKDCONLoss_r:100.0_a:1.0_b:0.8_0/resnet8x4_best.pth  2>&1 |tee log_rep/resnet8x4_STL10.log
