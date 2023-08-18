BASE_DIR=/home/$USER/code/ghost-noise
cd $BASE_DIR/submodules/timm

lr=1e-3
wd=1e-4

# Setting 1, resnet augmentation
for seed in 42 43 44
do
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output /home/$USER/runs/timm/ \
        --data-dir /home/$USER/datasets --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 --train-split "train" --val-split "test" \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model cifar_simple_vit \
        -b 1024 --opt adamw --lr $lr --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 40 --checkpoint-hist 1 --seed $seed \
        --amp --amp-dtype float16 \
        --log-wandb --wandb-kwargs project=nodo_svit name="\"final-c10-svit-b1-s$seed\""

    gbs=12
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output /home/$USER/runs/timm/ \
        --data-dir /home/$USER/datasets --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 --train-split "train" --val-split "test" \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model cifar_simple_vit --model-kwargs norm_cfg="[{\"type\":\"norm\",\"subtype\":\"ln\",\"dkwargs\":{\"affine\":True}},{\"type\":\"noise\",\"subtype\":\"ghost_noise_injector_rep\",\"dkwargs\":{\"batch_size\":$gbs,\"channel_last\":True}}]" \
        -b 1024 --opt adamw --lr $lr --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 40 --checkpoint-hist 1 --seed $seed \
        --amp --amp-dtype float16 \
        --log-wandb --wandb-kwargs project=nodo_svit name="\"final-c10-svit-b1-gbs$gbs-s$seed\""
done
