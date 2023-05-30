# Assume BASE_DIR is ../../
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname $(dirname $SCRIPT_DIR))
cd $BASE_DIR/submodules/timm

# Provide your own values here if needed (directories may need to exist)
data_dir=/home/$USER/datasets
out_dir=/home/$USER/runs/timm
wandb_project=TEST

lr=0.3
gbs=16
for seed in 42 43 44
do
    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' conv_cfg="{\"subtype\":\"wsconv2d\",\"dkwargs\":{\"gain\":True}}" norm_cfg="[{\"type\":\"norm\",\"subtype\":\"gn\",\"dkwargs\":{\"G\":1,\"affine\":False}},{\"type\":\"noise\",\"subtype\":\"ghost_noise_injector_rep\",\"dkwargs\":{\"batch_size\":$gbs}},{\"type\":\"gain\",\"subtype\":\"standard\"},{\"type\":\"bias\",\"subtype\":\"standard\"}]" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"c100-rn18-b256-test-wsln-gnir-gbs$gbs-s$seed\"" 'tags=["c100_rn18_b256_test_wsln_gnir"]' --seed $seed

    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' conv_cfg="{\"subtype\":\"wsconv2d\",\"dkwargs\":{\"gain\":True}}" norm_cfg="[{\"type\":\"norm\",\"subtype\":\"gn\",\"dkwargs\":{\"G\":1,\"affine\":True}}]" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"c100-rn18-b256-test-wsln-s$seed\"" 'tags=["c100_rn18_b256_test_wsln"]' --seed $seed

    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="[{\"type\":\"norm\",\"subtype\":\"gn\",\"dkwargs\":{\"G\":1,\"affine\":True}}]" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"c100-rn18-b256-test-ln-s$seed\"" 'tags=["c100_rn18_b256_test_ln"]' --seed $seed

    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
    --data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train" --val-split "test" \
    --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
    --test-transforms 'PILToTensor={}' \
    --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' conv_cfg="{\"subtype\":\"wsconv2d\",\"dkwargs\":{\"gain\":False}}" norm_cfg="[{\"type\":\"norm\",\"subtype\":\"gn\",\"dkwargs\":{\"G\":1,\"affine\":True}}]" \
    -b 256 --opt sgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=$wandb_project name="\"c100-rn18-b256-test-wsln-nogain-s$seed\"" 'tags=["c100_rn18_b256_test_wsln_nogain"]' --seed $seed
done
