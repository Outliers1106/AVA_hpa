import time
import os
import shutil
import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import ParameterTuple
import threading
from mindspore.train.callback import Callback
from mindspore.train.callback._callback import set_cur_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
#from mindspore.train.serialization import _exec_save_checkpoint, _save_graph
import mindspore.context as context


# class LossCallBack(Callback):
#     """
#         Monitor the loss in training.

#         If the loss is NAN or INF terminating training.

#         Note:
#             If per_print_times is 0 do not print loss.

#         Args:
#             per_print_times (int): Print loss every times. Default: 1.
#     """

#     def __init__(self, data_size, per_print_times=1):
#         super(LossCallBack, self).__init__()
#         if not isinstance(per_print_times, int) or per_print_times < 0:
#             raise ValueError("print_step must be int and >= 0.")
#         self._per_print_times = per_print_times
#         self._loss = 0
#         self.data_size = data_size
#         self.step_cnt = 0
#         self.loss_sum = 0

#     def epoch_begin(self, run_context):
#         self.epoch_time = time.time()

#     def epoch_end(self, run_context):
#         epoch_seconds = time.time() - self.epoch_time
#         self._per_step_seconds = epoch_seconds / self.data_size
#         self._loss = self.loss_sum / self.step_cnt
#         self.step_cnt = 0
#         self.loss_sum = 0

#     def get_loss(self):
#         return self._loss

#     def get_per_step_time(self):
#         return self._per_step_seconds

#     def step_end(self, run_context):
#         cb_params = run_context.original_args()
#         if not isinstance(cb_params.net_outputs, list):
#             loss = cb_params.net_outputs.asnumpy()
#         else:
#             loss = cb_params.net_outputs[0].asnumpy()

#         # cb_params.batch_num means : dataset_size / batch_size
#         cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
#         # print("cur_step_in_epoch:",cur_step_in_epoch,"step loss:",loss)
#         # if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
#         self.loss_sum += loss
#         self.step_cnt += 1
#         # print("check speed step{}".format(cur_step_in_epoch))


class LossCallBack(Callback):
    """
        Monitor the loss in training.

        If the loss is NAN or INF terminating training.

        Note:
            If per_print_times is 0 do not print loss.

        Args:
            per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, data_size, per_print_times=1, logger=None):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self.logger = logger
        self._per_print_times = per_print_times
        self._loss = 0
        self.data_size = data_size
        self.step_cnt = 0
        self.loss_sum = 0

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_seconds = time.time() - self.epoch_time
        self._per_step_seconds = epoch_seconds / self.data_size
        self._loss = self.loss_sum / self.step_cnt
        self.step_cnt = 0
        self.loss_sum = 0

        cb_params = run_context.original_args()
        epoch_idx = (cb_params.cur_step_num - 1) // cb_params.batch_num + 1

        self.logger.info("the {} epoch's resnet result: "
                         " training loss {},"
                         "training per step cost {:.2f} s, total_cost {:.2f} s".format(
            epoch_idx, self._loss, self._per_step_seconds, self._per_step_seconds * cb_params.batch_num))

    def get_loss(self):
        return self._loss

    def get_per_step_time(self):
        return self._per_step_seconds

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if not isinstance(cb_params.net_outputs, list):
            loss = cb_params.net_outputs.asnumpy()
        else:
            loss = cb_params.net_outputs[0].asnumpy()

        # cb_params.batch_num means : dataset_size / batch_size
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        print("cur_step_in_epoch:",cur_step_in_epoch,"step loss:",loss)
        # if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
        self.loss_sum += loss
        self.step_cnt += 1
        # print("check speed step{}".format(cur_step_in_epoch))


class ModelCheckpoint_0_5(ModelCheckpoint):
    def __init__(self, prefix='CKP', directory=None, config=None, epoch_all=200, monitor="acc", mode="max"):
        super(ModelCheckpoint_0_5, self).__init__(prefix, directory, config)
        self.cur_acc = 0
        self.best_acc = 0
        self.cur_epoch = 0
        self.epoch_all = epoch_all

    def set_acc(self, acc):
        self.cur_acc = acc

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def step_end(self, run_context):
        pass

    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print("cur epoch", self.cur_epoch)
        print("cur acc", self.cur_acc)
        print("cur best", self.best_acc)
        # save graph (only once)
        if not self._graph_saved:
            graph_file_name = os.path.join(self._directory, self._prefix + '-graph.meta')
            _save_graph(cb_params.train_network, graph_file_name)
            self._graph_saved = True
        self._save_ckpt(cb_params, self._config.model_type)

    def end(self, run_context):
        """
        Save the last checkpoint after training finished.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.cur_epoch != self.epoch_all:
            return
        cb_params = run_context.original_args()
        _to_save_last_ckpt = True
        self._save_ckpt(cb_params, self._config.model_type, _to_save_last_ckpt)

        from mindspore.parallel._cell_wrapper import destroy_allgather_cell
        destroy_allgather_cell()

    def _save_ckpt(self, cb_params, model_type, force_to_save=False):
        """Save checkpoint files."""

        step_num_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self.cur_acc > self.best_acc or force_to_save:
            print("epoch{} save".format(self.cur_epoch - 1))
            self.best_acc = self.cur_acc
            cur_ckpoint_file = self._prefix + "-" + str(self.cur_epoch - 1) + "_" \
                               + str(self.best_acc) + "_" + str(step_num_in_epoch) + ".ckpt"
            print("save path:{}".format(cur_ckpoint_file))
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
            elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
                self._cur_time_for_keep = time.time()
                if (self._cur_time_for_keep - self._last_time_for_keep) \
                        < self._config.keep_checkpoint_per_n_minutes * 60:
                    self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                               self._cur_time_for_keep)

            # generate the new checkpoint file and rename it.
            global _save_dir
            _save_dir = self._directory
            cur_file = os.path.join(self._directory, cur_ckpoint_file)
            tmp_ckpt_file_name_for_cur_process = str(os.getpid()) + "-" + 'parameters.ckpt'
            gen_file = os.path.join(_save_dir, tmp_ckpt_file_name_for_cur_process)
            self._last_time_for_keep = time.time()
            self._last_triggered_step = cb_params.cur_step_num

            if context.get_context("enable_ge"):
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()

            _exec_save_checkpoint(cb_params.train_network, gen_file, model_type, self._config.integrated_save)

            if os.path.exists(gen_file):
                shutil.move(gen_file, cur_file)
            self._latest_ckpt_file_name = cur_file


# 0.6 version
class ModelCheckpoint_0_6(ModelCheckpoint):
    def __init__(self, prefix='CKP', directory=None, config=None, epoch_all=200):
        super(ModelCheckpoint_0_6, self).__init__(prefix, directory, config)
        self.cur_acc = 0
        self.best_acc = 0
        self.cur_epoch = 0
        self.epoch_all = epoch_all

    def set_acc(self, acc):
        self.cur_acc = acc

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def step_end(self, run_context):
        pass

    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print("cur epoch", self.cur_epoch)
        print("cur acc", self.cur_acc)
        print("cur best", self.best_acc)
        # save graph (only once)
        if not self._graph_saved:
            graph_file_name = os.path.join(self._directory, self._prefix + '-graph.meta')
            _save_graph(cb_params.train_network, graph_file_name)
            self._graph_saved = True
        self._save_ckpt(cb_params)

    def end(self, run_context):
        """
        Save the last checkpoint after training finished.
        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.cur_epoch != self.epoch_all:
            return
        cb_params = run_context.original_args()
        _to_save_last_ckpt = True
        self._save_ckpt(cb_params, _to_save_last_ckpt)

        thread_list = threading.enumerate()
        if len(thread_list) > 1:
            for thread in thread_list:
                if thread.getName() == "asyn_save_ckpt":
                    thread.join()

        from mindspore.parallel._cell_wrapper import destroy_allgather_cell
        destroy_allgather_cell()

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""

        step_num_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self.cur_acc > self.best_acc or force_to_save:
            cur_epoch = self.cur_epoch
            if force_to_save:
                print("last epoch{} save".format(cur_epoch))
            else:
                cur_epoch = cur_epoch - 1
                print("epoch{} save".format(cur_epoch))
            self.best_acc = self.cur_acc
            cur_ckpoint_file = self._prefix + "-" + str(cur_epoch) + "_" \
                               + str(self.best_acc) + "_" + str(step_num_in_epoch) + ".ckpt"
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
            elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
                self._cur_time_for_keep = time.time()
                if (self._cur_time_for_keep - self._last_time_for_keep) \
                        < self._config.keep_checkpoint_per_n_minutes * 60:
                    self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                               self._cur_time_for_keep)

            # generate the new checkpoint file and rename it.
            global _save_dir
            _save_dir = self._directory
            cur_file = os.path.join(self._directory, cur_ckpoint_file)
            self._last_time_for_keep = time.time()
            self._last_triggered_step = cb_params.cur_step_num

            if context.get_context("enable_ge"):
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()

            _exec_save_checkpoint(cb_params.train_network, cur_file, self._config.integrated_save,
                                  self._config.async_save)

            self._latest_ckpt_file_name = cur_file
