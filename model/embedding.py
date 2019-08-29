import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class WordEmbedding(BaseModel):
    """Some Information about WordEmbedding"""

    def __init__(self, word_vec_mat, word_embedding_dim=50, add_unk_and_blank=True):
        super(WordEmbedding, self).__init__()
        assert (
            word_vec_mat.shape[1] == word_embedding_dim
        ), "word_embedding_dim is set as {} while the shape of input embedding is {}".format(
            word_embedding_dim, word_vec_mat.shape
        )
        pretrained_embeddings = torch.tensor(
            word_vec_mat, dtype=torch.float32
        )
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings , freeze=False, padding_idx = -1 if add_unk_and_blank else None
        )

    def forward(self, x):
        return self.embedding(x.long())


class PosEmbedding(BaseModel):
    """Some Information about PosEmbedding"""

    def __init__(self, pos_embedding_dim=5, max_length=120):
        super(PosEmbedding, self).__init__()
        pos_tot = max_length * 2
        self.pos1_embedding = nn.Embedding(
            num_embeddings=pos_tot, embedding_dim=pos_embedding_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=pos_tot, embedding_dim=pos_embedding_dim
        )
        nn.init.xavier_uniform_(self.pos1_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.pos2_embedding.weight, gain=1)
        

    def forward(self, pos1, pos2):
        pos1_v = self.pos1_embedding(pos1.long())
        pos2_v = self.pos2_embedding(pos2.long())
        x = torch.cat((pos1_v, pos2_v), -1)
        return x


class WordPosEmbedding(BaseModel):
    """Some Information about WordPosEmbedding"""

    def __init__(
        self,
        word_vec_mat,
        word_embedding_dim=50,
        add_unk_and_blank=True,
        pos_embedding_dim=5,
        max_length=120,
    ):
        super(WordPosEmbedding, self).__init__()
        self.word_embedding = WordEmbedding(
            word_vec_mat, word_embedding_dim, add_unk_and_blank
        )
        self.pos_embedding = PosEmbedding(pos_embedding_dim, max_length)

    def forward(self, word, pos1, pos2):
        word_v = self.word_embedding(word)
        pos_v = self.pos_embedding(pos1, pos2)
        x = torch.cat((word_v, pos_v), -1)
        return x
