export PYTHONPATH=/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
source /home/tuyanlun/code/mindspore_r1.0/env.sh

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_train.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../train.py \
    --device_id 5 \
    --device_num 1 \
    --device_target Ascend \
    --load_ckpt_path /home/tuyanlun/code/mindspore_r1.0/hpa/AVA-hpa-pretrain-resnet18-27-613/checkpoint-20210115-151444/AVA-100_3469.ckpt\
    --data_dir /home/tuyanlun/code/mindspore_r1.0/hpa_dataset/hpa \
    --save_checkpoint_path /home/tuyanlun/code/mindspore_r1.0/hpa \
    --log_path /home/tuyanlun/code/mindspore_r1.0/hpa