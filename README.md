# Source Code for the paper: 
# Calibration Transfer via Knowledge Distillation
Deep neural network (DNN) models have become increasingly prevalent in critical applications such as healthcare , and autonomous driving. In such applications, it is crucial for DNN predictions to not only be accurate but also trustworthy. Calibration refers to the alignment between a DNN model's predicted confidence and the actual frequency of the event it represents. Calibration indicates model's ability to provide reliable uncertainty estimates, and many modern DNNs are shown to be miscalibrated.

In this work, we explore if access to label uncertainties during training can prevent such overfitting and generate a calibrated classifier. Knowledge distillation (KD) has been used for transferring learned representations from a (typically large) teacher model to a (usually smaller) student model in the multitude of works. In this work, we investigate specifically if access to learned calibrated confidence through a teacher model also helps in the calibration of a student model.

## Setting Up Environment:
Run `setup_env.sh` file to set up your environment (creates a new Conda environment with name `kd`):
```
bash setup_env.sh
```


## Requirements:
We have tested this code on the following environment:  
- Python 3.8.16 / Pytorch (>=1.8.0) / torchvision (>=0.9.0) / CUDA 11.3

## Training:  
Please refer to the supplementary for hyperparameters.

### Training Teacher:  
To train a teacher model (say ResNet-56) on CIFAR-100 dataset on a single GPU-0 (say), run the following command:
```
CUDA_VISIBLE_DEVCES=0 python train_teacher.py --dataset cifar100 --model resnet56 --lr 0.1 --wd 5e-4 --train-batch-size 128 --loss <NLL or NLL+MDCA or FL or FL+MDCA> --gamma <FL loss weight hyperparameter> --beta <MDCA loss weight hyperparameter> --checkpoint calibrated_teachers/
```
<!--In our experiments, we chose `--gamma` from `{1, 2, 3, 4, 5}` and `--beta` from `{1, 5, 10}` for FL+MDCA.
For MMCE, `--beta` was chosen from `{1, 2, 3, 4, 5}`.  
For FL+MDCA, provide `--loss` as FL+MDCA and set `--beta` and `--gamma` both to your desired values.  
To train on label smoothing (LS), set `--loss` to `LS` and set `--alpha` to `0.1`.-->

### Training student:
To train a student model (say ResNet-8) using a pretrained teacher (ResNet-56) on CIFAR-100 dataset on GPU-0 (say), run the following command:
```
CUDA_VISIBLE_DEVICES=0 python train_student.py --dataset cifar100 --model resnet8 --teacher resnet56 --teacher_path {dir_path_to_the_ResNet-56_saved_model} --lr 0.1 --wd 5e-4 --train-batch-size 128 --Lambda <Distillation Weight> --T <Temperature> --checkpoint distilled_students/
```
You can plugin any values you like for `--T` (temperature $T$) and `--Lambda` (distillation weight or $\alpha$ in the paper) arguments.

### Multiple Model Training For Hypersearch:
To perform a hypersearch in Loss weight hyperparameters while training a teacher or in Temperature and Lambda while distilling a student, we use `simple_gpu_scheduler` library, available [here](https://github.com/ExpectationMax/simple_gpu_scheduler.git).  

For instance, to perform a grid search on Lambda and T:
```
simple_hypersearch "python train_student.py --dataset cifar100 --model resnet8 --teacher resnet56 --teacher_path {dir_path_to_the_ResNet-56_saved_model} --teacher_loss <loss_that_was_used_to_train_the_teacher> --lr 0.1 --wd 5e-4 --train-batch-size 128 --checkpoint distilled_model_library/ --T {T} --Lambda {Lambda}" -p T 1 1.5 2 3 4 5 10 20  -p Lambda 0.9 1.0 --sampling-mode grid | simple_gpu_scheduler --gpus 0,1,2,3
```
This command will schedule the training command inside "" (quotes) with different values of `--T` and `--Lambda` on GPUs with IDs: 0,1,2,3 in parallel.  

### Optional:
If you want to use [`accelerate`](https://github.com/huggingface/accelerate) library for multi GPU and fp16 training, make sure you run the command `accelerate config` once to setup [`accelerate`](https://github.com/huggingface/accelerate) default parameters after the environment setup.  

You can run models with accelerate library as follows:
```
CUDA_VISIBLE_DEVCES=0 accelerate launch --num_processes 1 --mixed_precision fp16  train_teacher.py --dataset cifar100 --model resnet56 --lr 0.1 --wd 5e-4 --train-batch-size 128 --loss cross_entropy --checkpoint calibrated_teachers/
```

Refer to more training commands in `scripts/` folder.
