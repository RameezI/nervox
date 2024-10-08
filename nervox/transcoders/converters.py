"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
import argparse
import tensorflow as tf

from nervox.utils import auxiliaries as aux

# def get_data_generator(params):
#     def get_metadata(dataset_opts):
#         return dataset_opts['metadata']
#
#     datasets_pkg = params['trainer']['datasets_pkg']
#     dataset_name = params['trainer']['dataset_name']
#     dataset_conf = params['dataset']
#     dataset = __import__('{}.{}'.format(datasets_pkg, dataset_name), fromlist=[dataset_name]).Dataset
#
#     # test if we can access to dataset, otherwise use the default download directory to download and prepare the dataset
#     dataset_exists = os.path.exists(dataset_conf['datasets_dir'])
#     dataset_conf.update({'datasets_dir': dataset_conf['datasets_dir'] if dataset_exists else None})
#     dataset_conf.update({'metadata': get_metadata(dataset_conf)})
#     ds = dataset(**dataset_conf)
#
#     def image_gen_tfrt():
#         image = np.random.normal(size=(1, 256, 256, 3)).astype(np.float32),
#         yield image
#
#     def image_gen_tflite():
#         ds_train, _ = ds(batch_size=1)
#         for i, (images, labels) in enumerate(ds_train):
#             yield [images]
#
#     return image_gen_tfrt


def convert_to_tfrt(run_dir, quantization_modes=("JIT_FLOAT16",)):
    valid_quantization_modes = {
        "JIT_FLOAT16",
        "JIT_FLOAT32",
        "INT8",
        "FLOAT16",
        "FLOAT32",
    }
    if quantization_modes is None:
        quantization_modes = valid_quantization_modes
    else:
        if isinstance(quantization_modes, str):
            quantization_modes = {quantization_modes}
        if isinstance(quantization_modes, tuple):
            quantization_modes = set(quantization_modes)
        assert isinstance(quantization_modes, set)
        assert quantization_modes.issubset(valid_quantization_modes)

    export_dir = os.path.join(run_dir, "export")

    if {"JIT_FLOAT16"}.issubset(quantization_modes):
        params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP16")
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=export_dir, conversion_params=params
        )
        converter.convert()
        converter.save(os.path.join(export_dir, "tensorRT", "jit_fp16"))

    if {"JIT_FLOAT32"}.issubset(quantization_modes):
        params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP32")
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=export_dir, conversion_params=params
        )
        converter.convert()
        converter.save(os.path.join(export_dir, "tensorRT", "jit_fp32"))

    if {"INT8", "FLOAT16", "FLOAT32"}.intersection(quantization_modes):
        params_file = os.path.join(run_dir, "params.json")
        assert os.path.isfile(
            params_file
        ), "parameters file, '{}' ,  does not exist".format(params_file)
        parameters = aux.read_params_from_file(params_file)
        data_generator = get_data_generator(parameters)

        if {"INT8"}.issubset(quantization_modes):
            params = tf.experimental.tensorrt.ConversionParams(
                precision_mode="INT8", maximum_cached_engines=1, use_calibration=True
            )
            converter = tf.experimental.tensorrt.Converter(
                input_saved_model_dir=export_dir, conversion_params=params
            )
            converter.convert(callibration_input_fn=data_generator)
            converter.save(os.path.join(export_dir, "tensorRT", "int8"))

        if {"FLOAT16"}.issubset(quantization_modes):
            params = tf.experimental.tensorrt.ConversionParams(
                precision_mode="FP16", maximum_cached_engines=16
            )
            converter = tf.experimental.tensorrt.Converter(
                input_saved_model_dir=export_dir, conversion_params=params
            )
            converter.convert()
            converter.build(
                input_fn=data_generator
            )  # Generate corresponding TRT engines
            converter.save(
                os.path.join(export_dir, "tensorRT", "float16")
            )  # Generated engines will be saved.

        if {"FLOAT32"}.issubset(quantization_modes):
            params = tf.experimental.tensorrt.ConversionParams(
                precision_mode="FP32", maximum_cached_engines=16
            )
            converter = tf.experimental.tensorrt.Converter(
                input_saved_model_dir=export_dir, conversion_params=params
            )
            converter.convert()
            # get representative data generator
            params_file = os.path.join(run_dir, "params.json")
            assert os.path.isfile(
                params_file
            ), "parameters file, '{}' ,  does not exist".format(params_file)
            parameters = aux.read_params_from_file(params_file)
            data_generator = get_data_generator(parameters)

            converter.build(
                input_fn=data_generator
            )  # Generate corresponding TRT engines
            converter.save(
                os.path.join(export_dir, "tensorRT", "float16")
            )  # Generated engines will be saved.


def convert_to_tflite(run_dir, quantization_mode=None, name="model"):
    valid_quantization_modes = ("DYNAMIC_INT8", "FIXED_INT8", "DYNAMIC_FLOAT16")

    if quantization_mode is None:
        quantization_mode = valid_quantization_modes
    else:
        assert quantization_mode in valid_quantization_modes

    export_dir = os.path.join(run_dir, "export")
    imported_model = tf.saved_model.load(export_dir)
    concrete_function = imported_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]

    if "DYNAMIC_INT8" in quantization_mode:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        _name = "{}_dynamic_int8_quant".format(name)
        converter.post_training_quantize = True
        tflite_model = converter.convert()
        with open(
            os.path.join(export_dir, "{}.tflite".format(_name)), "wb"
        ) as flat_buffer:
            flat_buffer.write(tflite_model)

    if "FIXED_INT8" in quantization_mode:
        # create converter
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        _name = "{}_fixed_int8_quant".format(name)  # converted_model_name
        # get representative data generator
        params_file = os.path.join(run_dir, "params.json")
        assert os.path.isfile(
            params_file
        ), "parameters file, '{}' ,  does not exist".format(params_file)
        parameters = aux.read_params_from_file(params_file)
        data_gen = get_data_generator(parameters)
        # convert the model with the fixed ranges (based on the representative dataset)
        converter.post_training_quantize = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = tf.lite.RepresentativeDataset(data_gen)
        tflite_model = converter.convert()
        with open(
            os.path.join(export_dir, "{}.tflite".format(_name)), "wb"
        ) as flat_buffer:
            flat_buffer.write(tflite_model)

    if "DYNAMIC_FLOAT16" in quantization_mode:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        _name = "{}_dynamic_float16_quant".format(name)
        converter.post_training_quantize = True
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(
            os.path.join(export_dir, "{}.tflite".format(_name)), "wb"
        ) as flat_buffer:
            flat_buffer.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command line arguments for the converter"
    )
    default_log_dir = os.path.join(os.path.expanduser("~"), "tensorflow_logs")
    default_name = os.path.join("food101", "densenet_pretrained")
    default_run_id = "20210520-1755"

    parser.add_argument(
        "--logs_dir",
        required=False,
        type=str,
        default=default_log_dir,
        help="path to the top level tensorflow logs directory",
    )
    parser.add_argument(
        "--name",
        required=False,
        type=str,
        default=default_name,
        help="name of the experiment directory that contains the runs",
    )
    parser.add_argument(
        "--run_id",
        required=False,
        type=str,
        default=default_run_id,
        help="name of the experiment run",
    )
    args = parser.parse_args()

    exp_run_dir = os.path.join(args.logs_dir, args.name, args.run_id)
    convert_to_tfrt(exp_run_dir)
