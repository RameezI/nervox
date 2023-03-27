"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
import argparse
import os
import json
from nervox.utils import auxiliaries as aux


class Module(tf.Module):
    def __init__(self, name=None, model=None, objective=None):
        super(Module, self).__init__(name=name)
        self.model = model
        self.test_metrics = objective.test_metrics
    
    @tf.function
    def __call__(self, x, training=True, mask=None):
        x = self.model(x, training=training, mask=mask)
        return x
    
    @tf.function
    def __call__(self, x, training=False, mask=None):
        x = self.model(x, training=training, mask=mask)
        return x
    
    @tf.function
    def test(self, x, ground_truth=None, mask=None):
        predictions = self.model(x, training=False, mask=mask)
        if ground_truth is None:
            [self.test_metrics[key](predictions) for key in self.test_metrics]
            test_metrics = {key: self.test_metrics[key].result() for key in self.test_metrics}
        else:
            [self.test_metrics[key](ground_truth, predictions) for key in self.test_metrics]
            test_metrics = {key: self.test_metrics[key].result() for key in self.test_metrics}
        return test_metrics
    
    @tf.function
    def debug(self, x):
        debug_method = getattr(self.model, "debug", None)
        y = self.model.debug(x, training=False) if debug_method else None
        return y


class Exporter:
    def __init__(self,
                 exp_run_dir=None,
                 model_name=None,
                 model_opts=None,
                 model_attributes=None,
                 dataset_opts=None,
                 objective_name=None,
                 objective_opts=None,
                 strategy_name=None,
                 models_pkg=None,
                 objectives_pkg=None
                 ):
        
        # TODO: remove reset_default_graph()?
        tf.compat.v1.reset_default_graph()
        models_pkg = 'nervox.models' if models_pkg is None else models_pkg
        model = __import__('{}.{}'.format(models_pkg, model_name), fromlist=[model_name]).Graph
        
        objectives_pkg = 'nervox.objectives' if objectives_pkg is None else objectives_pkg
        objective = __import__('{}.{}'.format(objectives_pkg, objective_name), fromlist=[objective_name]).Objective
        
        self.ckpt_dir = os.path.join(exp_run_dir, "checkpoints")
        self.export_dir = os.path.join(exp_run_dir, "export")
        
        self.model = model(**model_opts)
        objective_opts = dict() if objective_opts is None else objective_opts
        self.objective = objective(**objective_opts)
        
        if model_attributes is not None:
            self.model_attributes = {field: getattr(self.model, field, None) for field in model_attributes}
        else:
            self.model_attributes = None
        
        # create a  module for testing, predicting and re-training the model
        self.module = Module(name=model_name, model=self.model, objective=self.objective)
        
        # Analyse input output tensors for signature specification
        self.metadata = dataset_opts['metadata']
        input_keys = self.metadata.get('input_features', ('image',))
        output_keys = self.metadata.get('output_features', ('label',))
        
        input_names = [key for key in input_keys]
        input_shapes = [(None,) + tuple(self.metadata['shapes'][key]) for key in input_keys]
        input_tensor_specs = [tf.TensorSpec(shape, tf.float32, name)
                              for name, shape in zip(input_names, input_shapes)]
        
        output_names = [key for key in output_keys]
        output_shapes = [(None,) + tuple(self.metadata['shapes'][key]) for key in output_keys]
        out_tensor_specs = [tf.TensorSpec(shape, tf.float32, name)
                            for name, shape in zip(output_names, output_shapes)]
        
        # TODO: : analyse/test for various multi input/output scenarios
        input_tensor_specs = input_tensor_specs[0] if len(input_tenstor_specs) == 1 else input_tensor_specs
        out_tensor_specs = out_tensor_specs[0] if len(out_tensor_specs) == 1 else out_tensor_specs
        
        # test/predict/train/debug signatures for export
        predict_signature = self.module.__call__.get_concrete_function(x=input_tensor_specs, training=False)
        train_signature = self.module.__call__.get_concrete_function(x=input_tensor_specs, training=True)
        
        gt_spec = out_tensor_specs if strategy_name in ['explicit_supervision'] else None
        test_signature = self.module.test.get_concrete_function(x=input_tensor_specs, ground_truth=gt_spec)
        
        debug_signature = self.module.debug.get_concrete_function(x=input_tensor_specs)
        
        self.module_signatures = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature,
                                  'predict': predict_signature,
                                  'train': train_signature,
                                  'test': test_signature,
                                  }
        
        # optionally add a debug signature if the debug method is present
        self.module_signatures.update({'debug': debug_signature}) if debug_signature.outputs else None
        
    def save_model_attributes(self, export_path):
        file_path = os.path.join(export_path, 'assets', 'model_attributes.json')
        with open(file_path, 'w') as _file:
            json.dump(self.model_attributes, _file, sort_keys=True, indent=4)
    
    def save_metadata(self, export_path):
        file_path = os.path.join(export_path, 'assets', 'metadata.json')
        with open(file_path, 'w') as _file:
            json.dump(self.metadata, _file, sort_keys=True, indent=4)
    
    def __call__(self, export_dir):
        # First create directories
        os.makedirs(self.export_dir) if not os.path.isdir(self.export_dir) else None
        checkpoint = tf.train.Checkpoint(model=self.model)
        # TODO: remove expect_partial?
        status = checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_dir)).expect_partial()
        print(status)
        tf.saved_model.save(self.module, export_dir, self.module_signatures)
        self.save_metadata(export_dir) if self.metadata is not None else None
        self.save_model_attributes(export_dir) if self.model_attributes is not None else None


def export(exp_run_dir=None,
           model_updates=None,
           model_attributes=None
           ):
    params_file = os.path.join(exp_run_dir, 'params.json')
    assert os.path.isfile(params_file), ("The parameters file does not exist: ", params_file)
    params = aux.read_params_from_file(params_file)
    
    # update model conf if requested
    model_conf = params['model']
    model_conf.ClassificationMetricUpdater(model_updates) if model_updates is not None else None
    objective_conf = params['objective']
    
    tf_exporter = Exporter(exp_run_dir=exp_run_dir,
                           model_name=params['trainer']['model_name'],
                           model_opts=model_conf,
                           model_attributes=model_attributes,
                           objective_name=params['trainer']['objective_name'],
                           objective_opts=objective_conf,
                           strategy_name=params['trainer']['training_config']['strategy'],
                           dataset_opts=params['dataset'],
                           models_pkg=params['trainer']['models_pkg'],
                           objectives_pkg=params['trainer']['objectives_pkg'],
                           )
    
    export_dir = os.path.join(exp_run_dir, 'export')
    tf_exporter(export_dir)
    export_statement = "Model is exported, to analyse invoke:\n{}"
    print(export_statement.format("saved_model_cli show --dir {} --tag_set serve".format(export_dir)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command line arguments for the spinner")
    parser.add_argument('--exp_run_dir', type=str, help='path to the directory containing experiment parameters')
    args = parser.parse_args()
    run_dir = os.path.join(os.path.expanduser("~"), 'tensorflow_logs', 'cifar10', 'convnet', 'test')
    run_dir = run_dir if args.exp_run_dir is None else args.exp_run_dir
    export(exp_run_dir=run_dir)
