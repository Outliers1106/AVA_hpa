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

#from optimizer import SGD_ as SGD
import mindspore.dataset.engine as de
#from mindspore.train.callback import SummaryCollector

from src.config import get_train_config, save_config, get_logger
# from imagenet_dataset import get_train_dataset, get_test_dataset, get_train_test_dataset
from src.datasets import makeup_pretrain_dataset, makeup_dataset
# from config import get_config, save_config, get_logger
# from datasets import get_train_dataset, get_test_dataset, get_train_test_dataset

from src.resnet import resnet18, resnet50, resnet101
from src.network_define_train import WithLossCell, TrainOneStepCell
from src.network_define_eval import EvalCell,EvalMetric,EvalCallBack
from src.callbacks import LossCallBack
from src.loss import LossNet
from src.lr_schedule import step_cosine_lr, cosine_lr
from src.loss import BCELoss
#from knn_eval import KnnEval, FeatureCollectCell

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="AVA pretraining")
parser.add_argument("--device_id", type=int, default=1, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
#parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
#parser.add_argument("--mindspore_version", type=float, default=0.6, help="Mindspore version default 0.6.")
#parser.add_argument('--load_ckpt_path', type=str, default='/home/tuyanlun/code/mindspore_r1.0/hpa/AVA-hpa-resnet50/checkpoint-20201027-181404/AVA-27_2185.ckpt', help='checkpoint path of pretrain model')

args_opt = parser.parse_args()


if __name__ == '__main__':
    config = get_train_config()

    checkpoint_dir = config.checkpoint_dir
    data_dir = config.data_dir
    log_dir = config.log_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    device_id = args_opt.device_id
    device_num = args_opt.device_num

    # context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=device_id)

    print("device num:{}".format(device_num))
    print("device id:{}".format(device_id))

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=False, parameter_broadcast=True,full_batch=False)
        init()
        temp_path = os.path.join(temp_path, str(device_id))
        print("temp path with multi-device:{}".format(temp_path))

    # if args_opt.use_moxing:
    #     mox.file.shift('os', 'mox')
    #     mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=temp_path)
    #     #mox.file.copy_parallel(src_url=os.path.join(args_opt.data_url, 'val'), dst_url=os.path.join(temp_path, 'val'))

    #     checkpoint_dir = os.path.join(temp_path, config.moxing_save_checkpoint_path)
    #     train_data_dir = os.path.join(temp_path, config.moxing_train_data_dir)
    #     #test_data_dir = os.path.join(temp_path, config.moxing_test_data_dir)
    #     #log_dir = os.path.join(temp_path, config.moxing_log_dir)
    # else:
    #     checkpoint_dir = os.path.join(temp_path, config.save_checkpoint_path)
    #     train_data_dir = os.path.join(temp_path, config.train_data_dir)
    #     #test_data_dir = os.path.join(temp_path, config.test_data_dir)
    #     #log_dir = os.path.join(temp_path, config.log_dir)

    logger = get_logger(os.path.join(log_dir, 'log' + config.time_prefix + '.log'))

    print("start create dataset...")

    epoch_for_dataset = config.epochs
    
    train_dataset = makeup_dataset(data_dir=data_dir, mode='train', batch_size=config.batch_size_for_train, bag_size=config.bag_size_for_train, shuffle=True)
    eval_dataset = makeup_dataset(data_dir=data_dir, mode='val', batch_size=config.batch_size_for_eval, bag_size=config.bag_size_for_eval)
    train_dataset.__loop_size__ = 1
    eval_dataset.__loop_size__ = 1

    train_dataset_batch_num = int(train_dataset.get_dataset_size())
    eval_dataset_batch_num = int(eval_dataset.get_dataset_size())
    print("train dataset.get_dataset_size:{}".format(train_dataset.get_dataset_size()))
    print("eval dataset.get_dataset_size:{}".format(eval_dataset.get_dataset_size()))
    print("the chosen network is {}".format(config.network))
    logger.info("the chosen network is {}".format(config.network))

    if config.network == 'resnet18':
        resnet = resnet18(low_dims=config.low_dims, pretrain=False)
    elif config.network == 'resnet50':
        resnet = resnet50(low_dims=config.low_dims, pretrain=False)
    elif config.network == 'resnet101':
        resnet = resnet101(low_dims=config.low_dims, pretrain=False)
    else:
        raise ("Unsupported net work!")
    if config.load_ckpt:
        print("load checkpoint from {},{}".format(config.load_ckpt_path,config.load_ckpt_filename))
        load_checkpoint(os.path.join(config.load_ckpt_path, config.load_ckpt_filename), net=resnet)
    else:
        print("dont load checkpoint")
    # logger.info(resnet)

    loss = BCELoss(reduction='mean')

    net_with_loss = WithLossCell(resnet, loss)

    if config.lr_schedule == "cosine_lr":
        lr = Tensor(cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)
    else:
        lr = Tensor(step_cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            epoch_stage=config.epoch_stage,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)
    
    if config.type == 'SGD':
        opt = SGD(params=net_with_loss.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    elif config.type == 'Adam':
        opt = Adam(params=net_with_loss.trainable_params(), learning_rate=lr, beta1=config.beta1,
            beta2=config.beta2, weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if device_num > 1:
        net = TrainOneStepCell(net_with_loss, opt, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt)

    eval_network = EvalCell(resnet, loss)

    loss_cb = LossCallBack(data_size=train_dataset_batch_num,logger=logger)
    
    cb = [loss_cb]

    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * train_dataset_batch_num,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix='AVA', directory=checkpoint_dir, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net, metrics={'results_return': EvalMetric()},
                  eval_network=eval_network)
    #model._init(train_dataset,eval_dataset)
    epoch_per_eval = {"epoch":[], "f1_macro":[], "f1_micro":[], "auc":[], "val_loss":[]}
    
    eval_cb = EvalCallBack(model=model, eval_dataset=eval_dataset, eval_per_epoch=config.eval_per_epoch, 
                epoch_per_eval=epoch_per_eval, logger=logger)
    cb += [eval_cb]

    logger.info("save configs...")
    print("save configs...")
    # save current config
    config_name = 'config.json'
    save_config([os.path.join(checkpoint_dir, config_name)], config)

    logger.info("training begins...")
    print("training begins...")

    model.train(config.epochs, train_dataset, callbacks=cb,dataset_sink_mode=False)
    # model.train(config.epochs, dataset, callbacks=cb, dataset_sink_mode=True)
    # try:
    #     
    # except Exception as e:
    #     if device_num>1:
    #         obs_save_path=os.path.join(config.moxing_model_save_path, config.prefix)
    #         obs_save_path=os.path.join(obs_save_path,str(device_id))
    #         mox.file.copy_parallel(src_url=os.path.join(temp_path, config.prefix),
    #                                dst_url= obs_save_path)
    #     else:
    #         mox.file.copy_parallel(src_url=os.path.join(temp_path, config.prefix),
    #                            dst_url=os.path.join(config.moxing_model_save_path, config.prefix))

    # for epoch_idx in range(1, config.epochs + 1):
    #     # ckpoint_cb.set_epoch(epoch_idx)
    #     model.train(1, train_dataset, callbacks=cb, dataset_sink_mode=True)
        
    #     time_cost = loss_cb.get_per_step_time()
    #     loss = loss_cb.get_loss()
    #     print("the {} epoch's resnet result: "
    #           " training loss {}, "
    #           "training per step cost {:.2f} s, total_cost {:.2f} s".format(
    #         epoch_idx, loss, time_cost, time_cost * train_dataset_batch_num))
    #     logger.info("the {} epoch's resnet result: "
    #                 " training loss {},"
    #                 "training per step cost {:.2f} s, total_cost {:.2f} s".format(
    #         epoch_idx, loss, time_cost, time_cost * train_dataset_batch_num))

    #     if epoch_idx % config.eval_pause == 0:
    #         start = time.time()
    #         output = model.eval(eval_dataset)
    #         val_loss, lab_f1_macro, lab_f1_micro, lab_auc = output['results']

    #         end = time.time()
    #         logger.info("the {} epoch's Eval result: "
    #                 "eval loss {}, f1_macro {}, f1_micro {}, auc {},"
    #                 "eval cost {:.2f} s".format(
    #         epoch_idx, val_loss, lab_f1_macro, lab_f1_micro, lab_auc, end-start))

        

    # if args_opt.use_moxing:
    #     print("download file to obs...")
    #     if device_num>1:
    #         obs_save_path=os.path.join(config.moxing_model_save_path, config.prefix)
    #         obs_save_path=os.path.join(obs_save_path,str(device_id))
    #         mox.file.copy_parallel(src_url=os.path.join(temp_path, config.prefix),
    #                                dst_url= obs_save_path)

    #     else:
    #         mox.file.copy_parallel(src_url=os.path.join(temp_path, config.prefix),
    #                            dst_url=os.path.join(config.moxing_model_save_path, config.prefix))
