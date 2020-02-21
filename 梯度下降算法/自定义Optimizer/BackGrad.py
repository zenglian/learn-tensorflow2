#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-02-12 17:23
# @Author  : Leon Zeng
# @Link    : http://github.com/zenglian
# @Usage   : BackGrad
# @See https://github.com/openai/iaf/blob/master/tf_utils/adamax.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class BackGrad(optimizer_v2.OptimizerV2):
    """Optimizer that implements the BackGrad algorithm."""
    _bg_slots = {}

    """@@__init__"""

    def __init__(self, learning_rate=1.0, use_locking=False, name="BackGrad", **kwargs):
        """
        :param learning_rate: default stride of all variables of all layers
        :param use_locking: use blocking mode
        :param name: name of this optimizer
        """
        super(BackGrad, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(BackGrad, self)._prepare_local(var_device, var_dtype, apply_state)

    def _create_slots(self, var_list):
        pass

    def _resource_apply_dense(self, grad, var):
        if optimizer_v2._var_key(var) not in self._slots:
            self.add_slot(var, "grad", grad * 1)
            self.add_slot(var, "stride", tf.ones_like(var) * self._get_hyper("learning_rate"))

        last_grad = self.get_slot(var, "grad")
        last_stride = self.get_slot(var, "stride")
        turned = math_ops.less(last_grad * grad, 0)  # 掉头
        stride = tf.where(turned, last_stride * 0.5, last_stride)  # 梯度掉头，步幅减半

        delta = grad * stride
        max_stride = math_ops.reduce_max([math_ops.reduce_max(delta), -math_ops.reduce_min(delta)])
        # delta = tf.cond(tf.greater(max_stride, 10), lambda: delta * 10 / max_stride, lambda: delta)
        var_update = state_ops.assign_sub(var, delta)
        slot_stride = state_ops.assign(last_stride, stride)
        slot_grad = state_ops.assign(last_grad, grad)
        return control_flow_ops.group(*[var_update, slot_grad, slot_stride])

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(BackGrad, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate')
        })
        return config
