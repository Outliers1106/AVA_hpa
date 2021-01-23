#export PYTHONPATH=/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
#source /home/tuyanlun/code/mindspore_r1.0/env.sh

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_eval.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../eval.py \
    --device_id 0 \
    --device_num 1 \
    --device_target Ascend \
    --model_arch resnet18 \
    --classes 27 \
    --ckpt_path /home/tuyanlun/code/mindspore_r1.0/hpa/AVA-hpa-train-resnet18-27-613/checkpoint-20210115-182437/AVA-20_13877.ckpt\
    --data_dir /home/tuyanlun/code/mindspore_r1.0/hpa_dataset/hpa \
    --save_eval_path ./613_27_no_pretrain/