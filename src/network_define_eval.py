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
import eval_metrics

class EvalCell(nn.Cell):

    def __init__(self, network, loss):
        super(EvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self.criterion = loss
        


    def construct(self, data, label):
        outputs = self._network(data)

        val_predict = []
        cur = 0
        numpy_predict = outputs
        for i in range(len(label)):
            # 取均值
            # print("sample_predict: ", np.shape(numpy_predict[cur : cur + nslice[i]]))
            sample_bag_predict = np.mean(numpy_predict[cur: cur + nslice[i]], axis=0)
            # print("sample_bag_predict: ", np.shape(sample_bag_predict))
            cur = cur + nslice[i]
            val_predict.append(sample_bag_predict)

        
        val_predict = np.array(val_predict)
        # print("val_predict(torch) : ", np.shape(val_predict))
        # 计算loss
        loss = self.criterion(val_predict, gt)

        return val_predict, loss

#TODO 把torch改成numpy实现
    
class EvalMetric(nn.Metric):

    def __init__(self, batch_size):
        super(EvalMetric, self).__init__()
        self.clear()
        self.batch_size = batch_size

    
    def clear(self):
        self.total_loss = 0.0
        self.np_label = []
        self.np_pd = []
        self.np_score = []
        self.cnt = 0

    def update(self, *inputs):
        val_predict, loss = inputs
        self.cnt = self.cnt+1
        self.total_loss += loss
        # 保存中间结果   
        self.val_pd = eval_metrics.threshold_tensor_batch(val_predict)
        self.np_pd.append(val_pd)
        self.np_score.append(val_predict)
        self.np_label.append(gt)

    def eval(self):
        loss = self.total_loss / self.cnt
        lab_f1_macro, lab_f1_micro, lab_auc = eval_metrics.torch_metrics(np_label, np_pd, score=np_score)
        return loss, lab_f1_macro, lab_f1_micro, lab_auc


    
