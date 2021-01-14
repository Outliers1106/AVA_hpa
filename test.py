import os

os.system('export PYTHONPATH=/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe')
os.system('source /home/tuyanlun/code/mindspore_r1.0/env.sh')
import argparse
import random
import numpy as np
import time
import logging

import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import SGD, Adam

# from optimizer import SGD_ as SGD
import mindspore.dataset.engine as de
# from mindspore.train.callback import SummaryCollector

from src.config import get_train_config, save_config, get_logger
# from imagenet_dataset import get_train_dataset, get_test_dataset, get_train_test_dataset
from src.datasets import makeup_pretrain_dataset, makeup_dataset
# from config import get_config, save_config, get_logger
# from datasets import get_train_dataset, get_test_dataset, get_train_test_dataset

from src.resnet import resnet18, resnet50, resnet101
from src.network_define_train import WithLossCell, TrainOneStepCell
from src.network_define_eval import EvalCell, EvalMetric, EvalCallBack
from src.callbacks import LossCallBack
from src.loss import LossNet
from src.lr_schedule import step_cosine_lr, cosine_lr
from src.loss import BCELoss

# from knn_eval import KnnEval, FeatureCollectCell

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="AVA pretraining")
parser.add_argument("--device_id", type=int, default=7, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
# parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
# parser.add_argument("--mindspore_version", type=float, default=0.6, help="Mindspore version default 0.6.")
# parser.add_argument('--load_ckpt_path', type=str, default='/home/tuyanlun/code/mindspore_r1.0/hpa/AVA-hpa-resnet50/checkpoint-20201027-181404/AVA-27_2185.ckpt', help='checkpoint path of pretrain model')
# parser.add_argument('--ckpt_path', type=str,
#                     default='/home/tuyanlun/code/mindspore_r1.0/hpa/AVA-hpa-train-resnet18-27/checkpoint-20201223-145622/AVA-20_9313.ckpt',
#                     help='model checkpoint path')
parser.add_argument('--ckpt_path', type=str,
                    default='/home/tuyanlun/code/mindspore_r1.0/hpa/AVA-hpa-train-resnet18-27/checkpoint-20201223-145622/AVA-20_9313.ckpt',
                    help='model checkpoint path')
parser.add_argument("--model_arch", type=str, default="resnet18", help='model architecture')
parser.add_argument("--data_dir", type=str, default="/home/tuyanlun/code/mindspore_r1.0/hpa_dataset/hpa",
                    help='dataset path')
parser.add_argument("--classes", type=int, default=27, help='class number')

args_opt = parser.parse_args()

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=args_opt.device_id)
    ckpt_path = args_opt.ckpt_path
    data_dir = args_opt.data_dir

    if args_opt.model_arch == 'resnet18':
        resnet = resnet18(pretrain=False, classes=args_opt.classes)
    elif args_opt.model_arch == 'resnet50':
        resnet = resnet50(pretrain=False, classes=args_opt.classes)
    elif args_opt.model_arch == 'resnet101':
        resnet = resnet101(pretrain=False, classes=args_opt.classes)
    else:
        raise ("Unsupported net work!")

    param_dict = load_checkpoint(ckpt_path)
    # param_dict2 = load_checkpoint(ckpt_path, net=resnet)
    # for key in param_dict:
    #     l1=np.array(param_dict[key])
    #     l2=np.array(param_dict2[key])
    #     res = l1==l2
    #
    #     if 0 in res:
    #         print(key,param_dict[key]==param_dict2[key])

    # print("param_dict:{}".format(param_dict.keys()))
    load_param_into_net(resnet, param_dict)
    test_dataset = makeup_dataset(data_dir=data_dir, mode='test', batch_size=3, bag_size=20, classes=args_opt.classes, num_parallel_workers=4)
    test_dataset.__loop_size__ = 1

    test_dataset_batch_num = int(test_dataset.get_dataset_size())

    loss = BCELoss(reduction='mean')
    test_network = EvalCell(resnet, loss)
    model = Model(test_network, metrics={'results_return': EvalMetric()},
                  eval_network=test_network)
    result = model.eval(test_dataset)
    print(result)
    # print("f1_macro:{}, f1_micro:{}, auc:{}".format(f1_macro,f1_micro,auc))
