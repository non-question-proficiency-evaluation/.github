import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    """
    Dimension indices for tensor operations.
    
    Attributes:
        batch: Batch dimension index (0)
        seq: Sequence dimension index (1)
        feature: Feature dimension index (2)
    """
    batch = 0
    seq = 1
    feature = 2


class AKT(nn.Module):
    def __init__(self, n_question: int, n_pid: int, d_model: int, n_blocks: int,
                 kq_same: int, dropout: float, model_type: str, final_fc_dim: int = 512, 
                 n_heads: int = 8, d_ff: int = 2048, l2: float = 1e-5, separate_qa: bool = False):
        super().__init__()
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question+1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self) -> None:
        """
        Reset difficulty parameters to zero for Rasch models.
        
        This method initializes the difficulty parameter embeddings to zero,
        which is used in AKT-Rasch models to learn problem-specific difficulty.
        """
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data: torch.Tensor, qa_data: torch.Tensor, target: torch.Tensor, 
                pid_data: torch.Tensor = None) -> tuple:
        """
        Forward pass of the AKT model.
        
        Args:
            q_data (torch.Tensor): Question IDs tensor of shape (batch_size, seq_len).
            qa_data (torch.Tensor): Question-answer combined IDs tensor of shape (batch_size, seq_len).
            target (torch.Tensor): Target labels tensor of shape (batch_size, seq_len).
                Values: 1.0 for correct, 0.0 for incorrect, -1.0 for padding.
            pid_data (torch.Tensor, optional): Problem IDs tensor of shape (batch_size, seq_len).
                Required for Rasch models (when n_pid > 0). Defaults to None.
        
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): Total loss including binary cross-entropy and L2 regularization.
                - predictions (torch.Tensor): Predicted probabilities of shape (batch_size * seq_len,).
                - mask_sum (torch.Tensor): Number of valid (non-padding) elements.
        
        Note:
            The model uses batch-first format. Padding values in target should be -1.0 or less.
        """
        # Batch First
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            # BS, seqlen, d_model #f_(ct,rt)
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = (qa_data-q_data)//self.n_question  # rt
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(qa_data)+q_embed_data

        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct
            pid_embed_data = self.difficult_param(pid_data)  # uq
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct
            qa_embed_diff_data = self.qa_embed_diff(
                qa_data)  # f_(ct,rt) or #h_rt
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data)  # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum()+c_reg_loss, m(preds), mask.sum()


class Architecture(nn.Module):
    """
    Transformer architecture stack for AKT model.
    
    This class implements the encoder-decoder architecture using transformer blocks.
    The encoder processes question-answer pairs, while the decoder uses cross-attention
    to predict future performance.
    
    Args:
        n_question (int): Number of unique questions (used for model configuration).
        n_blocks (int): Number of stacked transformer blocks.
        d_model (int): Dimension of attention input/output.
        d_feature (int): Dimension of input in each multi-head attention part.
            Should equal d_model / n_heads.
        d_ff (int): Dimension of feed-forward network inside transformer blocks.
        n_heads (int): Number of attention heads. Must satisfy: n_heads * d_feature = d_model.
        dropout (float): Dropout rate for regularization.
        kq_same (bool): Whether query and key use the same weights.
        model_type (str): Type of model architecture ('akt').
    
    Attributes:
        blocks_1 (nn.ModuleList): Encoder blocks for processing question-answer pairs.
        blocks_2 (nn.ModuleList): Decoder blocks with alternating self-attention and cross-attention.
    """
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    """
    Basic transformer block implementing self-attention and feed-forward layers.
    
    This is a standard transformer block following the "Attention is All You Need" paper.
    It contains multi-head attention, layer normalization, and position-wise feed-forward
    networks with residual connections.
    
    Args:
        d_model (int): Model dimension (embedding size).
        d_feature (int): Feature dimension per attention head (d_model / n_heads).
        d_ff (int): Feed-forward network dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate for regularization.
        kq_same (bool): Whether query and key use the same projection weights.
    
    Attributes:
        masked_attn_head (MultiHeadAttention): Multi-head attention module.
        layer_norm1 (nn.LayerNorm): Layer normalization after attention.
        layer_norm2 (nn.LayerNorm): Layer normalization after feed-forward.
        linear1 (nn.Linear): First linear layer in feed-forward network.
        linear2 (nn.Linear): Second linear layer in feed-forward network.
    """
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask: int, query: torch.Tensor, key: torch.Tensor, 
                values: torch.Tensor, apply_pos: bool = True) -> torch.Tensor:
        """
        Forward pass through the transformer layer.
        
        Args:
            mask (int): Masking mode for attention.
                - 0: Can only attend to past values (causal masking).
                - 1: Can attend to current and past values.
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            values (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            apply_pos (bool, optional): Whether to apply position-wise feed-forward network.
                Defaults to True.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        
        Note:
            The layer applies:
            1. Multi-head attention with residual connection and layer norm.
            2. Position-wise feed-forward network (if apply_pos=True) with residual connection.
        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with learnable positional encoding.
    
    Implements scaled dot-product attention with multiple heads, following the
    transformer architecture. Includes learnable gamma parameters for positional
    encoding effects.
    
    Args:
        d_model (int): Model dimension (must be divisible by n_heads).
        d_feature (int): Feature dimension per head (d_model / n_heads).
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate applied to attention scores.
        kq_same (bool): Whether query and key use the same projection weights.
        bias (bool, optional): Whether to use bias in linear projections. Defaults to True.
    
    Attributes:
        v_linear (nn.Linear): Value projection layer.
        k_linear (nn.Linear): Key projection layer.
        q_linear (nn.Linear, optional): Query projection layer (if kq_same=False).
        out_proj (nn.Linear): Output projection layer.
        gammas (nn.Parameter): Learnable parameters for positional encoding effect.
    """
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize linear layer parameters using Xavier uniform initialization.
        
        This method initializes the weights of all projection layers and sets
        biases to zero for proper training initialization.
        """
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor, zero_pad: bool) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Attention mask of shape (batch_size, 1, seq_len, seq_len).
                True values indicate positions that can be attended to.
            zero_pad (bool): Whether to apply zero padding to the first position.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        
        Note:
            The attention mechanism includes:
            1. Linear projections for Q, K, V
            2. Scaled dot-product attention with positional encoding effects
            3. Concatenation of heads and final projection
        """
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, d_k: int,
              mask: torch.Tensor, dropout: nn.Dropout, zero_pad: bool, 
              gamma: torch.Tensor = None) -> torch.Tensor:
    """
    Compute scaled dot-product attention with positional encoding effects.
    
    This function implements the core attention mechanism with learnable positional
    encoding effects. It computes attention scores, applies positional effects based
    on distance, and returns the weighted sum of values.
    
    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_len, d_k).
        k (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_len, d_k).
        v (torch.Tensor): Value tensor of shape (batch_size, n_heads, seq_len, d_k).
        d_k (int): Dimension of key/query (used for scaling).
        mask (torch.Tensor): Attention mask of shape (batch_size, 1, seq_len, seq_len).
        dropout (nn.Dropout): Dropout layer to apply to attention scores.
        zero_pad (bool): Whether to zero-pad the first position.
        gamma (torch.Tensor, optional): Learnable gamma parameters for positional encoding.
            Shape: (n_heads, 1, 1). Defaults to None.
    
    Returns:
        torch.Tensor: Attention output of shape (batch_size, n_heads, seq_len, d_k).
    
    Note:
        The attention mechanism includes:
        1. Scaled dot-product: QK^T / sqrt(d_k)
        2. Positional encoding effect based on distance
        3. Softmax normalization
        4. Weighted sum of values
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for sequences.
    
    This class provides learnable positional encodings that are optimized during
    training, as opposed to fixed sinusoidal encodings.
    
    Args:
        d_model (int): Model dimension (embedding size).
        max_len (int, optional): Maximum sequence length. Defaults to 512.
    
    Attributes:
        weight (nn.Parameter): Learnable positional embeddings of shape (1, max_len, d_model).
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get positional embeddings for input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Positional embeddings of shape (1, seq_len, d_model).
        """
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embeddings following the original Transformer paper.
    
    This class provides fixed (non-learnable) positional encodings using sinusoidal
    functions, as described in "Attention is All You Need".
    
    Args:
        d_model (int): Model dimension (embedding size).
        max_len (int, optional): Maximum sequence length. Defaults to 512.
    
    Attributes:
        weight (nn.Parameter): Fixed positional embeddings of shape (1, max_len, d_model).
            This parameter is not trainable (requires_grad=False).
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get positional embeddings for input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Positional embeddings of shape (1, seq_len, d_model).
        """
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
