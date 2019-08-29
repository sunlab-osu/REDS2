import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel

import pdb


class BagAttention(BaseModel):
    """Some Information about BagAttention"""

    def __init__(self, relation_num, hidden_size, dropout_prob=1.0):
        super(BagAttention, self).__init__()
        self.relation_num = relation_num
        self.hidden_size = hidden_size
        self.relation_embedding = nn.Embedding(
            num_embeddings=relation_num, embedding_dim=hidden_size
        )
        nn.init.xavier_uniform_(self.relation_embedding.weight, gain=1.0)
        self.bias = nn.Parameter(torch.zeros(1, relation_num))
        self.dropout = nn.Dropout(dropout_prob)

    def _final_logit(self, bag_repr, is_train=True):
        """
        training: bag_repr [num_bag, hidden_size]
        testing: bag_repr [num_bag, r, hidden_size]
        """
        logits = torch.matmul(
            bag_repr, torch.transpose(self.relation_embedding.weight, 0, 1)
        )
        if is_train:
            logits += self.bias
        else:
            logits += self.bias.unsqueeze(1)
        return logits

    def forward(self, x, size, scope, relation=None, is_train=True):
        """
        x: encoded sentences [num_total_sentences, hidden_size]
        relation: relation is assigned during training [num_total_sentences]
        ---
        train:
        bagged_repr: [num_bag, hidden_size]
        final_logit: [num_bag, r]
        test:
        bagged_repr_per_r: [num_bag, r, hidden_size]
        final_logit: [num_bag, r, r]
        """
        size = size.cpu().numpy()
        if is_train:
            # pdb.set_trace()
            bagged_repr = []
            assert relation is not None, "relation must be given during training"
            # [num_total_sentences, hidden_size]
            relation_v = self.relation_embedding(relation)
            attention_logit = torch.sum(x * relation_v, 1)  # [num_total_sentencesg]
            for i, cur_size in enumerate(size):
                if cur_size == 1:
                    bagged_repr.append(x[scope[i, 0]])
                else:
                    tmp_x = torch.sum(
                        x[scope[i, 0] : scope[i, 1]]
                        * F.softmax(
                            attention_logit[scope[i, 0] : scope[i, 1]]
                        ).unsqueeze(1),
                        0,
                    )
                    bagged_repr.append(tmp_x)
            bagged_repr = torch.stack(bagged_repr)
            bagged_repr = self.dropout(bagged_repr)
            final_logit = self._final_logit(
                bagged_repr, is_train=is_train
            )  # [num_bag, r]
            return bagged_repr, final_logit, None
        else:
            bagged_repr_per_r = []
            attention_logit_per_r = torch.matmul(
                x, torch.transpose(self.relation_embedding.weight, 0, 1)
            )  # [num_total_sentences, r]
            for i, cur_size in enumerate(size):
                if cur_size == 1:
                    bagged_repr_per_r.append(
                        x[scope[i, 0]].unsqueeze(0).repeat(self.relation_num, 1)
                    )
                else:
                    tmp_x = torch.matmul(
                        torch.transpose(
                            F.softmax(
                                attention_logit_per_r[scope[i, 0] : scope[i, 1]], 0
                            ),
                            0,
                            1,
                        ),
                        x[scope[i, 0] : scope[i, 1]],
                    )
                    bagged_repr_per_r.append(tmp_x)
            bagged_repr_per_r = torch.stack(
                bagged_repr_per_r
            )  # [num_bag, r, hidden_size]
            final_logit = self._final_logit(bagged_repr_per_r, is_train=is_train)
            return bagged_repr_per_r, final_logit, attention_logit_per_r


class BagAttention_v1(BaseModel):
    """Some Information about BagAttention"""

    def __init__(self, relation_num, hidden_size, dropout_prob=1.0):
        super(BagAttention_v1, self).__init__()
        self.relation_num = relation_num
        self.hidden_size = hidden_size
        self.relation_embedding = nn.Embedding(
            num_embeddings=relation_num, embedding_dim=hidden_size
        )
        self.bias = nn.Parameter(torch.zeros(1, relation_num))
        self.dropout = nn.Dropout(dropout_prob)

    def _final_logit(self, bag_repr, is_train=True):
        """
        training: bag_repr [num_bag, hidden_size]
        testing: bag_repr [num_bag, r, hidden_size]
        """
        logits = torch.matmul(
            bag_repr, torch.transpose(self.relation_embedding.weight, 0, 1)
        )
        if is_train:
            logits += self.bias
        else:
            logits += self.bias.unsqueeze(1)
        return logits

    def forward(
        self, x, bag_id, in_bag_index, num_bag, max_bag, relation=None, is_train=True
    ):
        """
        x: encoded sentences [num_total_sentences, hidden_size]
        relation: relation is assigned during training [num_total_sentences]
        ---
        train:
        bagged_repr: [num_bag, hidden_size]
        final_logit: [num_bag, r]
        test:
        bagged_repr_per_r: [num_bag, r, hidden_size]
        final_logit: [num_bag, r, r]
        """
        index = bag_id * max_bag + in_bag_index
        bagged_x = torch.zeros(num_bag * max_bag, self.hidden_size, requires_grad=True)
        bagged_x = bagged_x.scatter(
            0, index.unsqueeze(1).repeat(1, self.hidden_size), x
        )
        bagged_x = bagged_x.reshape(
            num_bag, max_bag, self.hidden_size
        )  # [num_bag, max_bag, hidden_size]
        if is_train:
            assert relation is not None, "relation must be given during training"
            # [num_total_sentences, hidden_size]
            relation_v = self.relation_embedding(relation)
            attention_logit = torch.sum(x * relation_v, 1)  # [num_total_sentencesg]
            bagged_attention_logit = torch.zeros(num_bag * max_bag, requires_grad=True)
            bagged_attention_logit = bagged_attention_logit.scatter(
                0, index, attention_logit
            )
            bagged_attention_logit = bagged_attention_logit.reshape(num_bag, max_bag)
            attention_mask = (bagged_attention_logit != 0).float()
            attention_mask += (attention_mask.sum(1, keepdim=True) == 0).float()
            bagged_attention_weight = F.softmax(
                bagged_attention_logit + (1 - 1 / attention_mask), 1
            )
            bagged_repr = torch.sum(
                bagged_x * bagged_attention_weight.unsqueeze(-1), 1
            )  # [num_bag, hidden_size]
            bagged_repr = self.dropout(bagged_repr)
            final_logit = self._final_logit(
                bagged_repr, is_train=is_train
            )  # [num_bag, r]
            return bagged_repr, final_logit, None
        else:
            attention_logit_per_r = torch.matmul(
                x, torch.transpose(self.relation_embedding.weight, 0, 1)
            )  # [num_total_sentences, r]
            bagged_attention_logit_per_r = torch.zeros(
                num_bag * max_bag, self.relation_num, requires_grad=True
            )
            bagged_attention_logit_per_r = bagged_attention_logit_per_r.scatter(
                0,
                index.unsqueeze(1).repeat(1, self.relation_num),
                attention_logit_per_r,
            )
            bagged_attention_logit_per_r = bagged_attention_logit_per_r.reshape(
                num_bag, max_bag, self.relation_num
            )
            attention_mask = (bagged_attention_logit_per_r != 0).float()
            attention_mask += (attention_mask.sum(1, keepdim=True) == 0).float()
            bagged_attention_weight = F.softmax(
                bagged_attention_logit_per_r + (1 - 1 / attention_mask), 1
            )
            bagged_repr_per_r = torch.matmul(
                torch.transpose(bagged_attention_weight, 1, 2), bagged_x
            )  # [num_bag, r, hidden_size]
            final_logit = self._final_logit(bagged_repr_per_r, is_train=is_train)
            return bagged_repr_per_r, final_logit, attention_logit_per_r
