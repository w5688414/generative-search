# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import paddle
from modeling import BloomBiEncoderModel, LlamaBiEncoderModel
from paddlenlp.transformers import BloomConfig, LlamaConfig
from paddlenlp.utils.log import logger


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        required=True,
        help="The directory of model.",
    )
    parser.add_argument(
        "--model_type",
        choices=["bloom", "llama", "baichuan"],
        default="bloom",
        help="The model types",
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    return parser.parse_args()


def load_tp_params(tp_degree, path):
    tp_state_dict_list = []
    for tp in range(tp_degree):
        tp_state_dict = {}
        tp_name = os.path.join(path, f"model_state.tp{tp:0>2d}.pdparams")
        logger.info(f"loading {tp_name}")
        tmp = paddle.load(tp_name, return_numpy=True)
        for k, v in tmp.items():
            tp_state_dict[k] = v
        # model_state.tp00-00001-of-00002
        # tp_name = os.path.join(path, f"model_state.tp{tp:0>2d}-00001-of-00002.pdparams")
        # logger.info(f"loading {tp_name}")
        # tmp = paddle.load(tp_name, return_numpy=True)
        # for k, v in tmp.items():
        #     tp_state_dict[k] = v
        tp_state_dict_list.append(tp_state_dict)
    return tp_state_dict_list


def merge_tensor_parallel(model_class, state_dict_list, config) -> None:
    """the entry of converting config and converting model file

    Args:
        input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
        config (PretrainedConfig): the PretrainedConfig instance of model
    """
    name_action_mappings = model_class._get_tensor_parallel_mappings(
        config, is_split=False
    )
    state_keys_map = model_class._resolve_prefix_keys(
        name_action_mappings.keys(), state_dict_list[0].keys()
    )

    for k, v in state_keys_map.items():
        name_action_mappings[v] = name_action_mappings.pop(k)

    state_dict_to_save = {}
    for key in state_dict_list[0].keys():
        tensor = state_dict_list[0][key]
        if key in name_action_mappings:
            ret = [x[key] for x in state_dict_list]
            action = name_action_mappings.pop(key)
            tensor = action(ret)

        state_dict_to_save[key] = tensor

    if len(name_action_mappings) > 0:
        for x in name_action_mappings.keys():
            logger.warning(
                f"key <{x}> need to merge tensor parallel but we can't find in model state."
            )

    logger.info("Finally, we merging state dict to fellowing tensors.")
    for k, v in state_dict_to_save.items():
        logger.info(f"{k}, {v.shape}, {v.dtype}")

    return state_dict_to_save


def main():
    args = parse_arguments()
    paddle.set_device(args.device)
    if args.model_type == "bloom":
        config = BloomConfig.from_pretrained(args.model_name_or_path)
        model_class = BloomBiEncoderModel
    elif args.model_type in ["baichuan", "llama"]:
        config = LlamaConfig.from_pretrained(args.model_name_or_path)
        model_class = LlamaBiEncoderModel

    tp_state_dict_list = load_tp_params(4, args.model_name_or_path)
    state_dict_to_save = merge_tensor_parallel(
        model_class=model_class, state_dict_list=tp_state_dict_list, config=config
    )
    logger.info("Saving")
    paddle.save(
        state_dict_to_save,
        os.path.join(args.model_name_or_path, "model_state.pdparams"),
    )


if __name__ == "__main__":
    main()
