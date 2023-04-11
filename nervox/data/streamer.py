"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""
import os
import json
import numpy as np
import tensorflow as tf
from copy import copy, deepcopy
import importlib
import tensorflow_datasets as tfds
from typing import Union
from collections.abc import Iterable
from pathlib import Path
from nervox.utils import capture_params


class DataStream:
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @capture_params(apply_local_updates=True)
    def __init__(self,
                 dataset: Union[str, tf.data.Dataset, tf.distribute.DistributedDataset],
                 version: Union[str, None] = '',
                 split: str = 'train',
                 batch_size: Union[None, int] = None,
                 shuffle_files: bool = True,
                 download: bool = True,
                 transforms=None,
                 datasets_dir: Union[str, None, os.PathLike] = None,
                 manual_dir: Union[str, None] = None,
                 add_tfds_id:bool=True,
                 pkg: Union[str, None] = None,
                 **kwargs_tfds) -> None:
        """
        
        Args:
            dataset:
            version:
            split:
            batch_size:
            shuffle_files:
            download:
            transforms:
            datasets_dir:
            manual_dir:
            tfds_id: 
            pkg:
            **kwargs_tfds:
        """
        
        datasets_dir: str = str(Path(Path.home(), 'tensorflow_datasets')) \
            if datasets_dir is None else str(datasets_dir)
        
        self.split_name = split
        self._stream: Union[None, tf.data.Dataset] = None
        self._info: Union[None, tfds.core.dataset_info] = None
        self._batch_size = batch_size
        self._metadata = None
        self._cardinality = None
        
        tfds.core.lazy_imports()
        
        if isinstance(dataset, str):
            try:
                pkg = 'nervox' if pkg is None else pkg
                importlib.import_module(f'.datasets.{dataset}', package=pkg)
            except ImportError:
                if dataset not in tfds.list_builders():
                    raise ImportError(f'Import Failure!\n'
                                      f'Failed to import dataset: `{dataset}` from package: `{pkg}`.\n'
                                      f'Please make sure the tfds dataset directory exists.')
            
            read_config = kwargs_tfds.pop('read_config', tfds.ReadConfig())
            read_config.add_tfds_id = add_tfds_id

            self._stream, self._info = tfds.load(name=dataset + version, split=split,
                                                 data_dir=datasets_dir, batch_size=batch_size,
                                                 shuffle_files=shuffle_files,
                                                 download=download,
                                                 with_info=True,
                                                 read_config=read_config,
                                                 **kwargs_tfds)
        
        elif isinstance(dataset, (tf.data.Dataset, tf.distribute.DistributedDataset)):
            none_indexes = [k for element_spec in tf.nest.flatten(dataset.element_spec)
                            for k, dim in enumerate(element_spec.shape)
                            if dim is None]
            is_batched = True if none_indexes and len(set(none_indexes)) == 1 else False
            dataset = dataset.unbatch() if is_batched and batch_size is not None else dataset
            self._stream = dataset if batch_size is None else dataset.batch(batch_size)
        
        transforms = [] if transforms is None else transforms
        assert isinstance(transforms, Iterable), "The transforms must be iterable"
        self._set_transforms(transforms)
    
    def _set_transforms(self, transforms):
        for transform in transforms:
            self._stream = self._stream.apply(transform)
    
    def __iter__(self):
        return self._stream.__iter__()
    
    def __bool__(self):
        return self._stream is not None
    
    @property
    def info(self):
        return self._info
    
    @property
    def examples_count(self):
        examples_count = np.NaN
        if self.info is not None:
            examples_count = self._info.splits[self.split_name].num_examples
        return examples_count
    
    @property
    def cardinality(self):
        return self._cardinality if self._cardinality is not None else \
            tf.data.experimental.cardinality(self._stream)
    
    def __str__(self):
        config = getattr(self, 'params')
        ds_name = config['dataset'] \
            if isinstance(config['dataset'], str) else ''
        return ds_name
    
    def __len__(self):
        length = tf.data.experimental.cardinality(self._stream)
        return length
    
    def as_dataset(self):
        return self._stream
    
    @property
    def batch_size(self):
        return self._batch_size
    
    def distribute(self):
        """
        Distributes the dataset according to the active strategy and returns a shallow copy of the data stream
        Returns: A data stream
        """
        distribution_strategy = tf.distribute.get_strategy()
        assert self._stream is not None, f"Can not make a distributed data stream from {self._stream} object"
        distributed_data_stream = distribution_strategy.experimental_distribute_dataset(self._stream)
        data_stream = copy(self)
        data_stream._stream = distributed_data_stream
        return data_stream
    
    def to_json(self):
        params = copy(getattr(self, 'params', {}))
       
        # convert dataset to string.
        # TODO@Rameez: check, what happens if this is a tf.data.Dataset class
        # can we create a json paramerization from the dataset object?
        params.update({'dataset': str(params['dataset'])})
       
        # convert transforms to json serializable format.
        #  Only possible if parametrization is json serializable.
        transforms = [{'module': transform.__module__,
                       'class': type(transform).__name__,
                       'config': getattr(transform, 'params', {})
                       } for transform in params['transforms']]
        
        params.update({'transforms': transforms})
        return params
