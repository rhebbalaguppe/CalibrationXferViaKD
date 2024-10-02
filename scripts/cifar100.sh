CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision fp16 train_teacher.py \
--dataset cifar100 \
--model resnet56 \
--lr 0.1 \
--epochs 160 \
--scheduler multistep \
--schedule-steps 80 120 \
--lr-decay-factor 0.1 \
--wd 1e-4 \
--train-batch-size 128 \
--loss FL+MDCA --gamma 3.0 --beta 1.0

#####################

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision fp16 train_teacher.py \
--dataset cifar100 \
--model resnet56 \
--lr 0.1 \
--epochs 160 \
--scheduler multistep \
--schedule-steps 80 120 \
--lr-decay-factor 0.1 \
--wd 1e-4 \
--train-batch-size 128 \
--loss cross_entropy

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision fp16 train_teacher.py \
--dataset cifar100 \
--model resnet56 \
--lr 0.1 \
--epochs 160 \
--scheduler multistep \
--schedule-steps 80 120 \
--lr-decay-factor 0.1 \
--wd 1e-4 \
--train-batch-size 128 \
--loss LS --alpha 0.1