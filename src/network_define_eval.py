import time
import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import ParameterTuple
from mindspore.train.callback import Callback
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
import src.eval_metrics as eval_metrics

class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval, logger):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval
        self.logger = logger

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        epoch_idx = (cb_param.cur_step_num - 1) // cb_param.batch_num + 1
        if epoch_idx % self.eval_per_epoch == 0:
            start = time.time()
            output = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            print(output)
            #val_loss, lab_f1_macro, lab_f1_micro, lab_auc = output['results']
            val_loss, lab_f1_macro, lab_f1_micro, lab_auc = output['results_return']
            end = time.time()
            self.logger.info("the {} epoch's Eval result: "
                    "eval loss {}, f1_macro {}, f1_micro {}, auc {},"
                    "eval cost {:.2f} s".format(
            epoch_idx, val_loss, lab_f1_macro, lab_f1_micro, lab_auc, end-start))

            self.epoch_per_eval["epoch"].append(epoch_idx)
            self.epoch_per_eval["f1_macro"].append(lab_f1_macro)
            self.epoch_per_eval["f1_micro"].append(lab_f1_micro)
            self.epoch_per_eval["auc"].append(lab_auc)
            self.epoch_per_eval["val_loss"].append(val_loss)



class EvalCell(nn.Cell):

    def __init__(self, network, loss):
        super(EvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self.criterion = loss
        


    def construct(self, data, label, nslice):
        outputs = self._network(data)

        val_predict = []
        cur = 0
        # numpy_predict = outputs.asnumpy()
        # label = label.asnumpy()
        #nslice = nslice.asnumpy()
        print("output:",outputs)
        #print(outputs.dtype)
        print("label",label)
        #print(label.dtype)
        print("slices",nslice)
        #print(nslice.dtype)
        #for i in range(len(label)):
            # 取均值
            #print(i)
            # print("sample_predict: ", np.shape(numpy_predict[cur : cur + nslice[i]]))
            # sample_bag_predict = np.mean(numpy_predict[cur: cur + nslice[i]], axis=0)
            # print("sample_bag_predict: ", np.shape(sample_bag_predict))
            # cur = cur + nslice[i]
            # val_predict.append(sample_bag_predict)

        
        # val_predict = np.array(val_predict)
        # print("val_predict(shape) : ", np.shape(val_predict))
        # # 计算loss
        # loss = self.criterion(val_predict, label)

        #return val_predict, loss, label
        return outputs, label, nslice

#TODO 把torch改成numpy实现
#from mindspore.nn.metrics import Metric
class EvalMetric(nn.Metric):
#class EvalMetric(Metric):
    def __init__(self):
        super(EvalMetric, self).__init__()
        print("evalmetric init")
        self.clear()

    
    def clear(self):
        print("clear")
        self.total_loss = 0.0
        self.np_label = []
        self.np_pd = []
        self.np_score = []
        self.cnt = 0

    def update(self, *inputs):
        print("update")
        # val_predict, loss, label = inputs
        val_predict = []
        cur = 0
        numpy_predict = inputs[0].asnumpy()
        label = inputs[1].asnumpy()
        nslice = inputs[2].asnumpy()
        for i in range(len(label)):
            #取均值
            #print("sample_predict: ", np.shape(numpy_predict[int(cur) : int(cur + nslice[i])]))
            sample_bag_predict = np.mean(numpy_predict[int(cur): int(cur)+ nslice[i]], axis=0)
            # print("sample_bag_predict: ", np.shape(sample_bag_predict))
            cur = cur + nslice[i]
            val_predict.append(sample_bag_predict)


        self.cnt = self.cnt+1
        self.total_loss += 0
        # 保存中间结果   
        val_pd = eval_metrics.threshold_tensor_batch(val_predict)
        self.np_pd.append(val_pd)
        self.np_score.append(val_predict)
        self.np_label.append(label)

    def eval(self):
        loss = self.total_loss / self.cnt
        lab_f1_macro, lab_f1_micro, lab_auc = eval_metrics.np_metrics(self.np_label, self.np_pd, score=self.np_score)
        return loss, lab_f1_macro, lab_f1_micro, lab_auc
