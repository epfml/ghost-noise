# Assume BASE_DIR is ../../
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname $(dirname $SCRIPT_DIR))
cd $BASE_DIR/submodules/timm

# Provide your own values here if needed (directories may need to exist)
data_dir=/home/$USER/datasets
out_dir=/home/$USER/runs/timm
wandb_project=TEST

lr=0.2
wd=2e-4
for seed in 42 43 44
do
    # Batch Norm Baseline
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs name=cifar_rn20 norm_cfg="bn2d" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c10-rn20-b256-bn-$seed\"" 'tags=["test_c10_rn20_b256"]' --seed $seed

    # GBN
    for gbs in 16 32 64
    do
        PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
        --data-dir $data_dir --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model cifar_resnet --model-kwargs name=cifar_rn20 norm_cfg="{\"subtype\":\"gbn2d\",\"dkwargs\":{\"batch_size\":$gbs}}" \
        -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
        --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c10-rn20-b256-gbn$gbs-$seed\"" 'tags=["test_c10_rn20_b256"]' --seed $seed
    done

    # GNI
    for gbs in 16 32 64
    do
        PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
        --data-dir $data_dir --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model cifar_resnet --model-kwargs name=cifar_rn20 norm_cfg="[{\"type\":\"norm\",\"subtype\":\"bn2d\",\"dkwargs\":{\"affine\":False}},{\"type\":\"noise\",\"subtype\":\"ghost_noise_injector_rep\",\"dkwargs\":{\"batch_size\":$gbs}},{\"type\":\"gain\",\"subtype\":\"standard\"},{\"type\":\"bias\",\"subtype\":\"standard\"}]" \
        -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
        --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c10-rn20-b256-bn-gni$gbs-$seed\"" 'tags=["test_c10_rn20_b256"]' --seed $seed
    done
done
