import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import ParameterTuple
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
import mindspore.ops as ops
import mindspore

class WithLossCell(nn.Cell):
    """
        Wrap the network with loss function to compute loss.

        Args:
            backbone (Cell): The target network to wrap.
            loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.concat = P.Concat()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data1, data2, data3, label):
        data = self.concat((data1, data2, data3))
        feature1, feature2, feature3 = self._backbone(data)
        return self._loss_fn(feature1, feature2, feature3, label)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


class TrainOneStepCell(nn.Cell):
    """
        Network training package class.

        Append an optimizer to the training network after that the construct function
        can be called to create the backward graph.

        Args:
            net_with_loss (Cell): The training network with loss.
            optimizer (Cell): Optimizer for updating the weights.
            sens (Number): The adjust parameter. Default value is 1.0.
            reduce_flag (bool): The reduce flag. Default value is False.
            mean (bool): Allreduce method. Default value is False.
            degree (int): Device number. Default value is None.
    """

    def __init__(self, net_with_loss, optimizer, sens=1.0, reduce_flag=False, mean=False, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.net_with_loss = net_with_loss
        self.weights = ParameterTuple(net_with_loss.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=False)
        self.gradient_names = [param.name + ".gradient" for param in self.weights]

        ops.Print()(self.weights)
        ops.Print()(net_with_loss.untrainable_params())
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

        self.mean = P.ReduceMean()
        self.eye = P.Eye()
        self.sum_keep_dim = P.ReduceSum(keep_dims=True)

    def diag_part_new(self, input, batch_size):
        eye_matrix = self.eye(batch_size, batch_size, mindspore.float32)
        input = input * eye_matrix
        input = self.sum_keep_dim(input, 1)
        return input

    def construct(self, data1, data2, data3, label):
        weights = self.weights

        loss = self.net_with_loss(data1, data2, data3, label)
        grads = self.grad(self.net_with_loss, weights)(data1, data2, data3, label)
        ops.Print()(self.gradient_names[-1], self.mean(self.diag_part_new(grads[-1], 32)))
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))



class SGD_(nn.SGD):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.gradient_names = [param.name + ".gradient" for param in self.parameters]
        #ops.Print()(self.gradient_names)
        self.param_len = len(self.gradient_names)
        #ops.Print()(self.param_len)
        self.mean = P.ReduceMean()
        self.eye = P.Eye()
        self.sum_keep_dim = P.ReduceSum(keep_dims=True)

    def diag_part_new(self, input, batch_size):
        eye_matrix = self.eye(batch_size, batch_size, mindspore.float32)
        input = input * eye_matrix
        input = self.sum_keep_dim(input, 1)
        return input

    def construct(self, gradients):
        #ops.Print()(self.gradient_names[-1],self.mean(self.diag_part_new(gradients[-1],32)))
        # for i in range(self.param_len):
        #     ops.Print()(self.gradient_names[i],gradients[i])
        #ops.Print()(len(gradients),len(self.gradient_names))
        temp_param = self.parameters
        self.parameters = self.parameters[0:-1]
        success = self._original_construct(gradients[0:-1])
        self.parameters = temp_param
        return success