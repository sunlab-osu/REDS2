import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
import pdb


def get_importance(x, y):
    sim = torch.sum(x * y.unsqueeze(1), 2)
    sim = F.softmax(sim, 1)
    return sim


class CNNEncoder(BaseModel):
    """Some Information about CNNEncoder"""

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride_size,
        activation,
        dropout_prob=1.0,
    ):
        super(CNNEncoder, self).__init__()
        self.cnn_encoder = nn.Conv1d(input_size, hidden_size, kernel_size, stride_size)
        nn.init.xavier_uniform_(self.cnn_encoder.weight, gain=1)
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh()}
        self.activation = activations[activation]
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, is_train=True):
        x = x.permute(0, 2, 1)
        encoded_x = self.cnn_encoder(x)
        pooled_x = encoded_x.permute(0, 2, 1).max(1)[0]
        output_x = self.activation(pooled_x)
        output_x = self.dropout(output_x)
        return output_x


class PCNNEncoder(BaseModel):
    """Some Information about CNNEncoder"""

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride_size,
        activation,
        dropout_prob=1.0,
    ):
        super(PCNNEncoder, self).__init__()
        self.cnn_encoder = nn.Conv1d(
            input_size, hidden_size, kernel_size, stride_size, padding=1
        )
        nn.init.xavier_uniform_(self.cnn_encoder.weight, gain=1)
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh()}
        self.activation = activations[activation]
        self.dropout = nn.Dropout(dropout_prob)
        self.mask_embedding = nn.Embedding.from_pretrained(
            torch.tensor(
                [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=torch.float32
            )
        )

    def _piecewise_pooling(self, x, mask):
        mask = self.mask_embedding(mask.long())
        pooled_x = (mask.unsqueeze(2) + x.unsqueeze(3)).max(1)[0] - 100
        return pooled_x.reshape(-1, 3 * x.shape[2])

    def forward(self, x, mask, is_train=True):
        x = x.permute(0, 2, 1)
        encoded_x = self.cnn_encoder(x).permute(0, 2, 1)
        pooled_x = self._piecewise_pooling(encoded_x, mask)
        output_x = self.activation(pooled_x)
        output_x = self.dropout(output_x)
        return output_x
