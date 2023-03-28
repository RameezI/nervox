"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""
import tensorflow as tf
from typing import Union, Callable, Dict, Collection

from nervox.utils.distribution \
    import DistributedLossWrapper

from nervox.metrics import \
    Metric, Mean


class Objective(tf.Module):
    """This class defines a container for the training objective and associated metrics to measure its progress. The
     objective  class also describes the optimizer used to optimize used the loss function. This class also applies
     compensations for distribution of the batch and the loss function across multiple devices. For example, when
     reducing the losses across multiple devices using sum_over_batch_size, it ensures that the correct global
     batch_size is used and not the device-local batch_size. The motivation behind this abstraction is to enable
     users write loss/optimizer/metrics for a single device,without worrying about the distribution strategies,
     while this class applies the necessary adjustments.
    
    Examples:
    
    """
    
    def __init__(self, loss: tf.keras.losses.Loss, optimizer: tf.keras.optimizers.Optimizer, *,
                 metrics: Collection[Metric] = tuple(), name: str = 'objective'):
        super().__init__(name)
        self._optimizer = optimizer
        self._loss: DistributedLossWrapper = DistributedLossWrapper(loss)
        self._optimizer: tf.keras.optimizers.Optimizer = optimizer
        self._loss_metric = Mean(name='loss')
        self._metrics = list(metrics) + [self._loss_metric]
    
    @property
    def name(self):
        return self._name
    
    @property
    def metrics(self):
        return self._metrics
    
    @property
    def optimizer(self):
        """
        Returns:
        The optimizer for the composed loss object of the objective
        """
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        """
        Placeholder for upgrading/swapping the optimizer.
        Args:
            value:

        Returns:

        """
        raise NotImplementedError
    
    def compute_loss(self, *args, **kwargs) -> tf.Tensor:
        """
        Computes the overall loss and updates the average, across all batches, loss metric.
        The class is distribution friendly i.e. it takes into account the corrections required
        for computing and aggregating losses over multiple replicas.
    
        Args:
            *args:                      The arguments accepted by the loss object
            **kwargs:                   Additional keyword args thar must be passed on to the loss object
    
        Returns:
        
        """
        loss_value = self._loss(*args, **kwargs)
        self._loss_metric.update(loss_value)
        return loss_value
    
    def reset_optimizer(self, lr: float, scheduler: Union[None, Callable, Dict[int, float]]):
        """
        Resets all optimizer parameters and restarts the optimization with a
        given initial learning rate and a learning  schedule.
        Args:
            lr:         The initial learning rate of the optimizer
            scheduler:  The scheduled updates of the learning rate.
        Returns:
        """
    
    def update_metrics(self, *args, exclude: Collection[str] = ('loss',), **kwargs) -> None:
        metrics = filter(lambda x: True if x.name not in exclude else False, self.metrics)
        for metric in metrics:
            metric.update(*args, **kwargs)
        # map(lambda x: x.update(*args, **kwargs), metrics)
    


# class ObjectiveConfigurer(ABC):
#     @abstracmethod
#     def __call__(self) -> Union[Objective, Collection[Objective]]:
#         pass