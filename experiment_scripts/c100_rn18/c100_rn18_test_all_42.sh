# Assume BASE_DIR is ../../
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname $(dirname $SCRIPT_DIR))
cd $BASE_DIR/submodules/timm

# Provide your own values here if needed (directories may need to exist)
data_dir=/home/$USER/datasets
out_dir=/home/$USER/runs/timm
wandb_project=TEST

lr=0.3
for seed in 42 42 33
do

    # Batch Norm Baseline
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="bn2d" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-bn-$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed

    # GBN
    gbs=16
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="{\"subtype\":\"gbn2d\",\"dkwargs\":{\"batch_size\":$gbs}}" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-bn-gni-gbs$gbs-s$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed

    # GNI
    gbs=16
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="[{\"type\":\"norm\",\"subtype\":\"bn2d\",\"dkwargs\":{\"affine\":False}},{\"type\":\"noise\",\"subtype\":\"ghost_noise_injector_rep\",\"dkwargs\":{\"batch_size\":$gbs}},{\"type\":\"gain\",\"subtype\":\"standard\"},{\"type\":\"bias\",\"subtype\":\"standard\"}]" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-bn-gni-gbs$gbs-s$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed

    # GNI - Shift
    gbs=16
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="[{\"type\":\"norm\",\"subtype\":\"bn2d\",\"dkwargs\":{\"affine\":False}},{\"type\":\"noise\",\"subtype\":\"ghost_noise_injector_rep\",\"dkwargs\":{\"batch_size\":$gbs,\"scale_noise\":False}},{\"type\":\"gain\",\"subtype\":\"standard\"},{\"type\":\"bias\",\"subtype\":\"standard\"}]" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-bn-gni-shift-gbs$gbs-s$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed

    # GNI - Scale
    gbs=8
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="[{\"type\":\"norm\",\"subtype\":\"bn2d\",\"dkwargs\":{\"affine\":False}},{\"type\":\"noise\",\"subtype\":\"nat\",\"dkwargs\":{\"batch_size\":$gbs,\"scale_shift\":False,\"shift_noise\":False,\"scale_noise\":True}},{\"type\":\"gain\",\"subtype\":\"standard\"},{\"type\":\"bias\",\"subtype\":\"bias\"}]" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-bn-gni-scale-gbs$gbs-s$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed

    # EBN
    gbs=8
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="{\"subtype\":\"en2d\",\"dkwargs\":{\"batch_size\":$gbs}}" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-ebn-gbs$gbs-s$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed

    # XBN
    gbs=32
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="{\"subtype\":\"xbn2d\",\"dkwargs\":{\"batch_size\":$gbs}}" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-xbn-gbs$gbs-s$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed

    # GCDO
    p=0.2
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' block_dropout_cfg="{\"type\":\"noise\",\"subtype\":\"flexible_dropout\",\"dkwargs\":{\"drop_probability\":$p,\"noise_type\":\"gaussian\",\"channel_wise\":False,\"scaling_type\":\"mean\"}}" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"test-c100-rn18-b256-bn-gcdo-p$p-s$seed\"" 'tags=["test_c100_rn18_b256"]' --seed $seed
done
