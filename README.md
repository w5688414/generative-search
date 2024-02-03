# generative-search

## 安装

推荐安装gpu版本的[PaddlePalle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)，安装命令如下：

```
python -m pip install paddlepaddle-gpu==2.6.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
安装其他依赖：
```
pip install -r requirements.txt
```

## 运行

### 单卡训练

```
python train.py --do_train \
              --model_name_or_path bigscience/bloomz-7b1-mt \
              --output_dir ./checkpoints \
              --train_data ./data/toy_finetune_data.jsonl \
              --overwrite_output_dir \
              --fine_tune_type bitfit \
              --sentence_pooling_method weighted_mean \
              --num_train_epochs 10 \
              --per_device_train_batch_size 4 \
```

### 多卡训练

```
python -m paddle.distributed.launch --gpus "4,5,6,7" train.py \
        --do_train \
        --model_name_or_path bigscience/bloomz-7b1-mt  \
        --num_train_epochs 10 --per_device_train_batch_size 4 \
        --evaluation_strategy no \
        --save_steps 30 \
        --passage_max_len 300 \
        --bf16 --fp16_opt_level O2 --tensor_parallel_degree 4 \
        --logging_steps 50 --output_dir outputs \
        --sentence_pooling_method weighted_mean \
        --fine_tune_type bitfit \
        --overwrite_output_dir \
        --train_data ./data/toy_finetune_data.jsonl \
        --is_batch_negative True \
        --save_total_limit 10 

python merge_tp_params.py --model_name_or_path outputs/checkpoint-30
```

## 评估

评估脚本：

```
python tests/benckmark_test.py --model_type bloom \
                              --query_model outputs/checkpoint-30 \
                              --passage_model outputs/checkpoint-30 \
                              --query_max_length 64 \
                              --passage_max_length 512
```