echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_train.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../train.py \
    --device_id 1 \
    --device_num 1 \
    --device_target Ascend \
    --data_dir /home/tuyanlun/code/mindspore_r1.0/hpa_dataset/hpa \
    --save_checkpoint_path /home/tuyanlun/code/mindspore_r1.0/hpa/hpa_new_exp \
    --log_path /home/tuyanlun/code/mindspore_r1.0/hpa/hpa_new_exp \
    --load_ckpt_path /home/tuyanlun/code/mindspore_r1.0/hpa/hpa_new_exp/AVA-hpa-pretrain-new-exp/checkpoint-20210324-184521/AVA-40_2185.ckpt \
    --save_eval_path /home/tuyanlun/code/mindspore_r1.0/hpa/hpa_new_exp