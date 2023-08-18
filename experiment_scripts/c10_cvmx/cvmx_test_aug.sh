BASE_DIR=/home/$USER/code/ghost-noise
cd $BASE_DIR/submodules/timm

lr=2e-2
wd=5e-2

# Setting 1, resnet augmentation
for seed in 42 43 44
do
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output /home/$USER/runs/timm/ --checkpoint-hist 1 \
        --model convmixer_cifar --model-kwargs kernel_size=5 patch_size=2 \
        --data-dir /home/$USER/datasets --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 --train-split "train" --val-split "test" \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        -b 256 --opt adamw --lr $lr --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-epochs 5 --seed $seed --opt-eps=1e-3 \
        --log-wandb --wandb-kwargs project=nodo_cvmx name="\"test-c10-cvmx-BG2-b1-lr$lr-seed$seed\"" 

    gbs=128
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output /home/$USER/runs/timm/ --checkpoint-hist 1 \
        --model convmixer_cifar --model-kwargs kernel_size=5 patch_size=2 norm_cfg="[{\"type\":\"norm\",\"subtype\":\"bn2d\",\"dkwargs\":{\"affine\":False}},{\"type\":\"noise\",\"subtype\":\"ghost_noise_injector_rep\",\"dkwargs\":{\"batch_size\":$gbs}},{\"type\":\"gain\",\"subtype\":\"standard\"},{\"type\":\"bias\",\"subtype\":\"standard\"}]" \
        --data-dir /home/$USER/datasets --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 --train-split "train" --val-split "test" \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        -b 256 --opt adamw --lr $lr --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-epochs 5 --seed $seed --opt-eps=1e-3 \
        --log-wandb --wandb-kwargs project=nodo_cvmx name="\"test-c10-cvmx-BG2-b1-gbs$gbs-seed$seed\"" 
done