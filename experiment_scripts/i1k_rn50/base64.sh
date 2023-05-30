# Assume BASE_DIR is ../../
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname $(dirname $SCRIPT_DIR))
cd $BASE_DIR/submodules/timm

# Provide your own values here if needed (directories may need to exist)
data_dir=/home/$USER/datasets
out_dir=/home/$USER/runs/timm
wandb_project=TEST

seed=0
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $out_dir \
--data-dir $data_dir/imagenet --dataset ImageFolder --num-classes 1000 --pin-mem --input-size 3 224 224 --workers 24 \
--train-transforms 'RandomResizedCrop=(224,)' 'RandomHorizontalFlip=(0.5,)' PILToTensor='{}' \
--test-transforms 'Resize=(256,)' 'CenterCrop=(224,)' 'PILToTensor={}' \
--model resnet50 --amp --channels-last \
-b 64 --local-accumulation 4 --opt sgd --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --sched cosine --sched-on-update --epochs 90 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 1 --checkpoint-hist 5 \
--log-wandb --wandb-kwargs project=$wandb_project name=rn50_i1k_sgd_base64x4_${seed} --seed $seed
