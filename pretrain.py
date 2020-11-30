import os
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
from mindspore.nn import SGD
#from optimizer import SGD_ as SGD
import mindspore.dataset.engine as de
#from mindspore.train.callback import SummaryCollector

from src.config import get_pretrain_config, save_config, get_logger
# from imagenet_dataset import get_train_dataset, get_test_dataset, get_train_test_dataset
from src.datasets import makeup_pretrain_dataset, makeup_dataset
# from config import get_config, save_config, get_logger
# from datasets import get_train_dataset, get_test_dataset, get_train_test_dataset

from src.resnet import resnet18, resnet50, resnet101
from src.network_define_pretrain import WithLossCell, TrainOneStepCell
from src.callbacks import LossCallBack
from src.loss import LossNet
from src.lr_schedule import step_cosine_lr, cosine_lr
#from knn_eval import KnnEval, FeatureCollectCell

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="AVA pretraining")
#parser.add_argument("--use_moxing", type=bool, default=False, help="whether use moxing for huawei cloud.")
#parser.add_argument("--data_url", type=str, default='', help="huawei cloud ModelArts need it.")
#parser.add_argument("--train_url", type=str, default='', help="huawei cloud ModelArts need it.")
#parser.add_argument("--src_url", type=str, default='obs://tuyanlun/data/', help="huawei cloud ModelArts need it.")
#parser.add_argument("--dst_url", type=str, default='/cache/data', help="huawei cloud ModelArts need it.")
#parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default is false.")
#parser.add_argument("--do_train", type=bool, default=True, help="Do train or not, default is true.")
#parser.add_argument("--do_eval", type=bool, default=False, help="Do eval or not, default is false.")
#parser.add_argument("--pre_trained", type=str, default="", help="Pretrain file path.")
parser.add_argument("--device_id", type=int, default=1, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
#parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
#parser.add_argument("--mindspore_version", type=float, default=0.6, help="Mindspore version default 0.6.")
args_opt = parser.parse_args()



if __name__ == '__main__':
    config = get_pretrain_config()

    checkpoint_dir = config.checkpoint_dir
    data_dir = config.data_dir
    log_dir = config.log_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # print("args_opt.data_url:", args_opt.data_url)
    # print("args_opt.train_url:", args_opt.train_url)
    # print("args_opt.src_url:", args_opt.src_url)
    # print("args_opt.dst_url:", args_opt.dst_url)
    #temp_path = ''
    # if args_opt.use_moxing:
    #     device_id = int(os.getenv('DEVICE_ID'))
    #     device_num = int(os.getenv('RANK_SIZE'))
    #     print("get path mapping with huawei cloud...")
    #     import moxing as mox

    #     temp_path = args_opt.dst_url

    print("do not use moxing for huawei cloud...")
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

    # epoch_for_dataset = config.epochs if args_opt.mindspore_version == 0.5 else 1
    epoch_for_dataset = config.epochs

    dataset = makeup_pretrain_dataset(data_dir=data_dir, batch_size=config.batch_size, bag_size=config.bag_size, shuffle=True)
    # dataset.__loop_size__ = 1

    # train_dataset = get_train_dataset(train_data_dir=train_data_dir, batchsize=config.batch_size,
    #                                   epoch=epoch_for_dataset, device_id=device_id, device_num=device_num)

    # for data in train_dataset.create_dict_iterator():
    #     print("train data:",data)
    #     break

    #train_dataset.__loop_size__ = 1

    # eval_dataset contains train dataset and test dataset, which is used for knn eval
    # eval_dataset = get_train_test_dataset(train_data_dir=train_data_dir, test_data_dir=test_data_dir,
    #                                       batchsize=100, epoch=epoch_for_dataset)

    # for data in eval_dataset.create_dict_iterator():
    #     print("eavl data:",data['image'])
    #     break

    dataset_batch_num = int(dataset.get_dataset_size())
    print("dataset.get_dataset_size:{}".format(dataset.get_dataset_size()))
    # eval_dataset_batch_num = int(eval_dataset.get_dataset_size())

    print("the chosen network is {}".format(config.network))
    logger.info("the chosen network is {}".format(config.network))

    if config.network == 'resnet18':
        resnet = resnet18(low_dims=config.low_dims, pretrain=True)
    elif config.network == 'resnet50':
        resnet = resnet50(low_dims=config.low_dims, pretrain=True)
    elif config.network == 'resnet101':
        resnet = resnet101(low_dims=config.low_dims, pretrain=True)
    else:
        raise ("Unsupported net work!")

    # logger.info(resnet)

    loss = LossNet(temp=config.sigma)

    net_with_loss = WithLossCell(resnet, loss)

    if config.lr_schedule == "cosine_lr":
        lr = Tensor(cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            steps_per_epoch=dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)
    else:
        lr = Tensor(step_cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            epoch_stage=config.epoch_stage,
            steps_per_epoch=dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)

    opt = SGD(params=net_with_loss.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if device_num > 1:
        net = TrainOneStepCell(net_with_loss, opt, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt)

    # eval_network = FeatureCollectCell(resnet)

    loss_cb = LossCallBack(data_size=dataset_batch_num,logger=logger)
    
    cb = [loss_cb]

    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_batch_num,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        # if args_opt.mindspore_version == 0.5:
        #     ckpoint_cb = ModelCheckpoint_0_5(prefix='AVA',
        #                                      directory=checkpoint_dir,
        #                                      config=ckptconfig)
        # elif args_opt.mindspore_version == 0.6:
        #     ckpoint_cb = ModelCheckpoint_0_6(prefix='AVA',
        #                                      directory=checkpoint_dir,
        #                                      config=ckptconfig)
        ckpoint_cb = ModelCheckpoint(prefix='AVA', directory=checkpoint_dir, config=ckptconfig)
        cb += [ckpoint_cb]

    # model = Model(net, metrics={'knn_acc': KnnEval(batch_size=config.batch_size, device_num=1)},
    #               eval_network=eval_network)
    model = Model(net)
    # model.init(train_dataset, eval_dataset)

    logger.info("save configs...")
    print("save configs...")
    # save current config
    config_name = 'config.json'
    save_config([os.path.join(checkpoint_dir, config_name)], config)

    logger.info("training begins...")
    print("training begins...")

    model.train(config.epochs, dataset, callbacks=cb, dataset_sink_mode=True)
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
    #     # output = model.eval(eval_dataset)
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
