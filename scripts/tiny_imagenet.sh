# train resnet on tiny_imagenet using warmup_cosine lr scheduling
# currently 352 steps per epoch on 4 GPUs using 64 batch-size per GPU

# train resnet on tiny_imagenet using warmup_cosine lr scheduling
# currently 352 steps per epoch on 4 GPUs using 64 batch-size per GPU
# define the number of GPUs to use in `num_processes`

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --mixed_precision fp16 train_teacher.py \
--dataset tiny_imagenet \
--model resnet110 \
--lr 0.1 \
--scheduler warmupcosine \
--warmup 1000 \
--wd 1e-4 \
--train-batch-size 64 \
--epochs 100 \
--loss cross_entropy

# not the change in batch size when using 1 GPU only
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision fp16 train_teacher.py \
--dataset tiny_imagenet \
--model mobilenetv2 \
--lr 0.1 \
--scheduler warmupcosine \
--warmup 1000 \
--wd 1e-4 \
--train-batch-size 256 \
--epochs 100 \
--loss cross_entropy

# train student resnet18
# can do this on single GPU
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision fp16 train_student.py \
--dataset tiny_imagenet \
--model resnet18 \
--lr 0.1 \
--scheduler warmupcosine \
--warmup 1000 \
--wd 1e-4 \
--train-batch-size 256 \
--epochs 100 \
--Lambda 0.9 \
--T 2.0 \
--teacher resnet110 \
--teacher_path checkpoint/tiny_imagenet/resnet110/2023-02-27-13:50:11.307690_cross_entropy

# posthoc calibrate
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 --num_processes 1 posthoc_calibrate.py --dataset tiny_imagenet --model resnet18 --lr 0.001 --patience 5 --checkpoint checkpoint/tiny_imagenet/resnet18/2023-03-02-05:57:32.128341_student_T=20.0_Lambda=0.1_runid=78