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

from dataclasses import dataclass
from typing import Dict, List, Optional

import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddlenlp.transformers import (AutoConfig, AutoModel, BloomConfig,
                                    LlamaConfig, LlamaModel,
                                    LlamaPretrainedModel, PretrainedModel)
from paddlenlp.transformers.bloom.modeling import (BloomModel,
                                                   BloomPreTrainedModel)
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.utils.log import logger


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[paddle.Tensor] = None
    p_reps: Optional[paddle.Tensor] = None
    loss: Optional[paddle.Tensor] = None
    scores: Optional[paddle.Tensor] = None


class BloomBiEncoderModel(BloomPreTrainedModel):
    def __init__(
        self,
        config: BloomConfig,
        normalized: bool = False,
        sentence_pooling_method: str = "cls",
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        is_batch_negative: bool = False,
        margin: float = 0.3,
    ):
        super().__init__(config)
        self.config = config
        self.bloom = BloomModel(config)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.is_batch_negative = is_batch_negative
        self.margin = margin
        if not normalized:
            self.temperature = 1.0
            logger.info(
                "reset temperature = 1.0 due to using inner product to compute similarity"
            )

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.labels = paddle.arange(0, self.config.seq_length, dtype="int64")
        self.labels.stop_gradient = True

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "weighted_mean":
            # Use weighted mean to compute similarity for decoder only LLMs
            # refer to https://github.com/Muennighoff/sgpt/blob/9728de441b1dd2e638a8a64e1c83f77716f47d9a/biencoder/beir/beir_dense_retriever.py#L258
            # 1,2,3...seq_len
            weights = (
                self.labels[1 : hidden_state.shape[1] + 1]
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(hidden_state.shape)
            )
            # [batch_size, seq_len] -> [batch_size, seq_len, higgen_dim]
            input_mask_expanded = mask.unsqueeze(-1).expand(hidden_state.shape)
            # bs, seq_len, hidden_dim -> bs, hidden_dim
            sum_embeddings = paddle.sum(
                hidden_state * input_mask_expanded * weights, axis=1, dtype="float32"
            )
            sum_mask = paddle.sum(input_mask_expanded * weights, axis=1)
            embedding = sum_embeddings / sum_mask
            return embedding

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.bloom(**features, return_dict=True)
        p_reps = self.sentence_embedding(
            psg_out.last_hidden_state, features["attention_mask"]
        )
        if self.normalized:
            p_reps = paddle.nn.functional.normalize(p_reps, axis=-1)
        return p_reps

    def compute_similarity(self, q_reps, p_reps):
        # q_reps [batch_size, embedding_dim]
        # p_reps [batch_size, embedding_dim]
        return paddle.matmul(q_reps, p_reps.transpose([1, 0]))

    def forward(
        self,
        inputs: Dict[str, paddle.Tensor] = None,
        teacher_score: paddle.Tensor = None,
    ):
        query = inputs["query"]
        passage = inputs["passage"]
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            if self.is_batch_negative:
                # In batch negatives
                scores = self.compute_similarity(q_reps, p_reps)
                # Substract margin from all positive samples cosine_sim()
                margin_diag = paddle.full(
                    shape=[q_reps.shape[0]], fill_value=self.margin, dtype=q_reps.dtype
                )
                scores = scores - paddle.diag(margin_diag)
                # Scale cosine to ease training converge
                scores = scores / self.temperature
                # 0,1,2,3...batch_size-1
                target = self.labels[0 : q_reps.shape[0]]
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.reshape([q_reps.shape[0], -1])
                # 0,1,2,3...batch_size-1
                target = self.labels[0 : scores.shape[0]]
                target = target * (p_reps.shape[0] // q_reps.shape[0])
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[paddle.Tensor]):
        if t is None:
            return None

        all_tensors = [paddle.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = paddle.concat(all_tensors, axis=0)

        return all_tensors


class LlamaBiEncoderModel(LlamaPretrainedModel):
    def __init__(
        self,
        config: LlamaConfig,
        normalized: bool = False,
        sentence_pooling_method: str = "cls",
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        is_batch_negative: bool = False,
        margin: float = 0.3,
    ):
        super().__init__(config)
        self.config = config
        self.llama = LlamaModel(config)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.is_batch_negative = is_batch_negative
        self.margin = margin
        if not normalized:
            self.temperature = 1.0
            logger.info(
                "reset temperature = 1.0 due to using inner product to compute similarity"
            )

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            if config.tensor_parallel_degree > 1:
                raise ValueError(
                    "Tensor parallelism does not support cross batch negatives."
                )

            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "weighted_mean":
            # Use weighted mean to compute similarity for decoder only LLMs
            # refer to https://github.com/Muennighoff/sgpt/blob/9728de441b1dd2e638a8a64e1c83f77716f47d9a/biencoder/beir/beir_dense_retriever.py#L258
            weights = (
                paddle.arange(start=1, end=hidden_state.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(hidden_state.shape)
            )
            # [batch_size, seq_len] -> [batch_size, seq_len, higgen_dim]
            input_mask_expanded = mask.unsqueeze(-1).expand(hidden_state.shape)
            # bs, seq_len, hidden_dim -> bs, hidden_dim
            sum_embeddings = paddle.sum(
                hidden_state * input_mask_expanded * weights, axis=1, dtype="float32"
            )
            sum_mask = paddle.sum(input_mask_expanded * weights, axis=1)
            embedding = sum_embeddings / sum_mask
        else:
            # TODO(wugaosheng): Add lasttoken pooling method
            raise NotImplementedError

        return embedding

    def encode(self, features):
        psg_out = self.llama(**features, return_dict=True)
        p_reps = self.sentence_embedding(
            psg_out.last_hidden_state, features["attention_mask"]
        )
        if self.normalized:
            p_reps = paddle.nn.functional.normalize(p_reps, axis=-1)
        return p_reps

    def compute_similarity(self, q_reps, p_reps):
        # q_reps [batch_size, embedding_dim]
        # p_reps [batch_size, embedding_dim]
        return paddle.matmul(q_reps, p_reps.transpose([1, 0]))

    def forward(
        self,
        inputs: Dict[str, paddle.Tensor] = None,
        teacher_score: paddle.Tensor = None,
    ):
        query = inputs["query"]
        passage = inputs["passage"]
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            if self.is_batch_negative:
                # In batch negatives
                scores = self.compute_similarity(q_reps, p_reps)
                # Substract margin from all positive samples cosine_sim()
                margin_diag = paddle.full(
                    shape=[q_reps.shape[0]], fill_value=self.margin, dtype=q_reps.dtype
                )
                scores = scores - paddle.diag(margin_diag)
                # Scale cosine to ease training converge
                scores = scores / self.temperature
                target = paddle.arange(0, q_reps.shape[0], dtype="int64")
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.reshape([q_reps.shape[0], -1])

                target = paddle.arange(scores.shape[0], dtype="int64")
                target = target * (p_reps.shape[0] // q_reps.shape[0])
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[paddle.Tensor]):
        if t is None:
            return None

        all_tensors = [paddle.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = paddle.concat(all_tensors, axis=0)

        return all_tensors


class BiEncoderModel(PretrainedModel):
    def __init__(
        self,
        model_name_or_path: str = None,
        normalized: bool = False,
        sentence_pooling_method: str = "cls",
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        use_inbatch_neg: bool = True,
        margin: float = 0.3,
        matryoshka_dims: Optional[List[int]] = None,
        matryoshka_loss_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model_config
        self.margin = margin
        self.matryoshka_dims = matryoshka_dims

        if self.matryoshka_dims:
            self.matryoshka_loss_weights = (
                matryoshka_loss_weights
                if matryoshka_loss_weights
                else [1] * len(self.matryoshka_dims)
            )
        else:
            self.matryoshka_loss_weights = None

        if not normalized:
            self.temperature = 1.0
            logger.info(
                "reset temperature = 1.0 due to using inner product to compute similarity"
            )

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "mean":
            s = paddle.sum(hidden_state * mask.unsqueeze(-1).float(), axis=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]

    def get_model_config(
        self,
    ):
        return self.model_config.to_dict()

    def encode(self, features):
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(
            psg_out.last_hidden_state, features["attention_mask"]
        )
        return p_reps

    def compute_similarity(self, q_reps, p_reps):
        # q_reps [batch_size, embedding_dim]
        # p_reps [batch_size, embedding_dim]
        return paddle.matmul(q_reps, p_reps.transpose([1, 0]))

    def forward(
        self,
        inputs: Dict[str, paddle.Tensor] = None,
        teacher_score: paddle.Tensor = None,
    ):
        query = inputs["query"]
        passage = inputs["passage"]
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            # Cross device negatives
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            if self.matryoshka_dims:
                loss = 0.0
                for loss_weight, dim in zip(
                    self.matryoshka_loss_weights, self.matryoshka_dims
                ):
                    reduced_q = q_reps[:, :dim]
                    reduced_d = p_reps[:, :dim]
                    if self.normalized:
                        reduced_q = paddle.nn.functional.normalize(reduced_q, axis=-1)
                        reduced_d = paddle.nn.functional.normalize(reduced_d, axis=-1)
                    scores = self.compute_similarity(reduced_q, reduced_d)
                    scores = scores / self.temperature
                    scores = scores.reshape([q_reps.shape[0], -1])

                    target = paddle.arange(scores.shape[0], dtype="int64")
                    target = target * (p_reps.shape[0] // q_reps.shape[0])
                    dim_loss = self.compute_loss(scores, target)
                    loss += loss_weight * dim_loss

            elif self.use_inbatch_neg:
                if self.normalized:
                    q_reps = paddle.nn.functional.normalize(q_reps, axis=-1)
                    p_reps = paddle.nn.functional.normalize(p_reps, axis=-1)
                # In batch negatives
                scores = self.compute_similarity(q_reps, p_reps)
                # Substract margin from all positive samples cosine_sim()
                margin_diag = paddle.full(
                    shape=[q_reps.shape[0]], fill_value=self.margin, dtype=q_reps.dtype
                )
                scores = scores - paddle.diag(margin_diag)
                # Scale cosine to ease training converge
                scores = scores / self.temperature
                target = paddle.arange(0, q_reps.shape[0], dtype="int64")
                loss = self.compute_loss(scores, target)
            else:
                if self.normalized:
                    q_reps = paddle.nn.functional.normalize(q_reps, axis=-1)
                    p_reps = paddle.nn.functional.normalize(p_reps, axis=-1)
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.reshape([q_reps.shape[0], -1])

                target = paddle.arange(scores.shape[0], dtype="int64")
                target = target * (p_reps.shape[0] // q_reps.shape[0])
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[paddle.Tensor]):
        if t is None:
            return None

        all_tensors = [paddle.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = paddle.concat(all_tensors, axis=0)

        return all_tensors

    @classmethod
    def from_pretrained(cls, **kwargs):
        # Instantiate model.
        model = cls(**kwargs)
        return model

    def save_pretrained(self, output_dir: str, **kwargs):
        # is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)
