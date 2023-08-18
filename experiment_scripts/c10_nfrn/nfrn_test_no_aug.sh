BASE_DIR=/home/$USER/code/ghost-noise
cd $BASE_DIR/submodules/timm

lr=2e-1
wd=5e-5

# Setting 0, no augmentation
for seed in 42 43 44
do
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output /home/$USER/runs/timm/ \
        --data-dir /home/$USER/datasets --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 --train-split "train" --val-split "test" \
        --train-transforms 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model cifar_nf_resnet26 --model-kwargs skipinit=True \
        -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 20 --checkpoint-hist 1 --seed $seed \
        --amp --amp-dtype float16 \
        --log-wandb --wandb-kwargs project=nodo_nfrn name="\"test-c10-nfrn-b0-s$seed\""

    gbs=32
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output /home/$USER/runs/timm/ \
        --data-dir /home/$USER/datasets --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 --train-split "train" --val-split "test" \
        --train-transforms 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model cifar_nf_resnet26 --model-kwargs skipinit=True noise_cfg="{\"type\":\"noise\",\"subtype\":\"ghost_noise_injector_rep\",\"dkwargs\":{\"batch_size\":$gbs}}" \
        -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 20 --checkpoint-hist 1 --seed $seed \
        --amp --amp-dtype float16 \
        --log-wandb --wandb-kwargs project=nodo_nfrn name="\"test-c10-nfrn-b0-gbs$gbs-s$seed\""
done