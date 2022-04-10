import torch
from torch import nn as nn
import math
from functools import partial


class Embedding(nn.Module):

    """
    This class converts the 3D data to 2D

     Parameters
    ----------
     seq_len : 
         The window_size of the data 

     embed_dim:
         Embedding dimension of the model which is considered as the number of electrodes in the transformer part

     grid:
         The size of the electrode grid    
   """

    def __init__(self):
        super().__init__()

        # self.embed=nn.Linear(seq_len,embed_dim)

    def forward(self, x):
        
        """
        x is (n_samples,seq_len,h_grid,v_grid)
        seq_lent: temporal sequnece length
        seq_lens: spatial sequence length
        
        """
        x = x.flatten(2)  # (n_samples,seq_lent,seq_lens)

        return x


class PositionalEncoding(nn.Module):

    """
    This class computes the sinusoid positional encoding matrix and adds it to the data 

     Parameters
    ----------

     embed_dim:
         Embedding dimension of the model which is considered as the number of electrodes in the transformer part

     max_len:
         Maximum length of the sequence (window_size) used for positional embedding
   """

    def __init__(self, embed_dim, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)  # (max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0)
        self.register_buffer('pe', pe)  # (max_len,embed_dim)

    def forward(self, x):

        seq_lent = x.size(1)
        seq_lens = x.size(2)
        x = x + self.pe.T[:seq_lent, :]  # x is (n_samples,seq_lent,seq_lens)

        return x


class MultiHeadAttention(nn.Module):

    """
    This class performs the Multi Head Attention mechanism on the data

     Parameters
    ----------

     embed_dim:
         Embedding dimension of the model which is considered as the number of electrodes in the transformer part

     n_heads:
         Number of heads in the attention part

     attn_drop:
         Dropout probability used in the Attention block after multiplying query and key    

     proj_p:
         Dropout probability used in the Attention and MLP blocks

    """

    def __init__(self, embed_dim, n_heads, attn_drop=0, proj_drop=0):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        n_samples, seq_lent, seq_lens = x.shape

        if seq_lens != self.embed_dim:
            raise ValueError

        qkv = self.qkv(x)     # (n_samples, seq_lent, 3 * seq_lens)
        qkv = qkv.reshape(
            n_samples, seq_lent, 3, self.n_heads, self.head_dim
        )  # (n_samples, seq_lent, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )             # (3, n_samples, n_heads, seq_lent, head_dim)

        # (n_samples, n_heads, seq_lent, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # (n_samples, n_heads, head_dim, seq_lent)
        k_t = k.transpose(-2, -1)
        dp = (
            q @ k_t
        ) * self.scale        # (n_samples, n_heads, seq_lent, seq_lent)

        # (n_samples, n_heads, seq_lent, seq_lent)
        attn = dp.softmax(dim=-1)

        attn = self.attn_drop(attn)
        # (n_samples, n_heads, seq_lent, head_dim)
        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(
            1, 2
        )                # (n_samples, seq_lent, n_heads, head_dim)
        # (n_samples, seq_lent, seq_lent)
        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)       # (n_samples, seq_lent, seq_lens)
        x = self.proj_drop(x)       # (n_samples, seq_lent, seq_lens)

        return x


class LSTMNet(nn.Module):

    """
    This class takes the data from the transforner encoder and applies a LSTM model on it

    Parameters
   ----------

   seq_lens:
       Spatial sequence length

   seq_lent:
       Temporal sequnec length
    """

    def __init__(self, seq_lens, seq_lent):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=seq_lens, hidden_size=seq_lens, num_layers=1, batch_first=True)

    def forward(self, x):

        x, _ = self.lstm(x)  # (n_samples, seq_lent, seq_lens)

        return x


class MLP(nn.Module):

    """
    This class correspnds to the MLP layer after attention in the tranformer encoder

    Parameters
   ----------

   in_features:
       Embedding dimension of the model which is the number of channels in this project

   hidden_features:
       mlp_ratio*in_features

   out_features:
       Embedding dimension

   proj_drop: 
       Dropout probability used in the Attention and MLP blocks
    """

    def __init__(self, in_features, hidden_features, out_features, proj_drop=0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x):

        x = self.fc1(x)     # (n_samples, seq_lent, hidden_features)
        x = self.act(x)     # (n_samples, seq_lent, hidden_features)
        x = self.drop(x)    # (n_samples, seq_lent, hidden_features)
        x = self.fc2(x)     # (n_samples, seq_lent, embed_dim or seq_lens)
        x = self.drop(x)    # (n_samples, seq_lent, seq_lens)

        return x


class Block(nn.Module):

    """
    This class return the transformer block by concatenating all the transformer's modules together

    Parameters
   ----------

   embed_dim:
       Embedding dimension of the model which is considered as the number of electrodes in the transformer part

   n_heads:
       Number of heads in the attention part

   mlp_ratio:
       The ratio of hidden_size/input_size in the MLP of the encoder

   attn_p:
       Dropout probability used in the Attention block after multiplying query and key    

   proj_drop: 
       Dropout probability used in the Attention and MLP blocks
    """

    def __init__(self, embed_dim, n_heads, mlp_ratio=4, attn_drop=0, proj_drop=0, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        self.norm2 = norm_layer(embed_dim, eps=1e-6)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=hidden_features,
            out_features=embed_dim,
            proj_drop=proj_drop
        )

    def forward(self, x):

        x = x + self.attn(self.norm1(x))  # (n_samples,seq_lent,seq_lens)
        x = x + self.mlp(self.norm2(x))

        return x


class Transformer(nn.Module):

    """
    The whole model in which all the above blocks are concatenated

    Parameters
   ----------

    seq_len:
        The window_size of the data

    max_len:
        Maximum length of the sequence (window_size) used for positional embedding

    grid:
        The size of the electrode grid

    n_classes:
        Number of gestures

    embed_dim:
        Embedding dimension of the model which is considered as the number of electrodes in the transformer part

    depth:
        Model's depth---The number of transforemr encoders that are serially concatenated to each other

    n_heads:
        Number of heads in the attention part

    mlp_ratio:
        The ratio of hidden_size/input_size in the MLP of the encoder

    p:
       Dropout probability used in the Attention and MLP blocks

    attn_p:
        Dropout probability used in the Attention block after multiplying query and key

    norm_layer:
        The normalization layer

    """

    def __init__(self, seq_len,
                 max_len,
                 n_classes,
                 embed_dim,
                 depth,
                 grid,
                 n_heads,
                 mlp_ratio,
                 p,
                 attn_p,
                 norm_layer
                 ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.flat = Embedding()

        self.pos_enc = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_len
        )

        self.pos_drop = nn.Dropout(p=p)

        self.lstm = LSTMNet(
            seq_lens=grid[0]*grid[1],
            seq_lent=seq_len
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim,
                  n_heads=n_heads,
                  mlp_ratio=mlp_ratio,
                  attn_drop=attn_p,
                  proj_drop=p,
                  norm_layer=norm_layer
                  )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(grid[0]*grid[1], n_classes)

    def forward(self, x):
        """
        seq_lens=grid[0]*grid[1]:
            Spatial sequence length

        seq_lent=seq_len:
            Temporal sequnec length

        Returns
        -------
        logits : torch.Tensor
            logits over all the classes `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]  # batch_size
        x = self.flat(x)  # (n_samples, seq_lent, seq_lens)
        x = self.pos_enc(x)  # (n_samples, seq_lent, seq_lens)
        x = self.pos_drop(x)  # (n_samples, seq_lent, seq_lens)

        for block in self.blocks:
            x = block(x)  # (n_samples, seq_lent, seq_lens)

        x = self.norm(x)  # (n_samples, seq_lent, seq_lens)
        x = self.lstm(x)  # (n_samples, seq_lent, seq_lens)
        f_state = x[:, -1, :]  # (n_samples, seq_lens)
        x = self.head(f_state)  # (n_samples, n_classes)

        return x
