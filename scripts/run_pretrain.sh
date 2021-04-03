echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_pretrain.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../pretrain.py \
    --device_id 1 \
    --device_num 1 \
    --device_target Ascend \
    --data_dir /home/tuyanlun/code/mindspore_r1.0/hpa_dataset/hpa \
    --save_checkpoint_path /home/tuyanlun/code/mindspore_r1.0/hpa/hpa_new_exp \
    --log_path /home/tuyanlun/code/mindspore_r1.0/hpa/hpa_new_exp