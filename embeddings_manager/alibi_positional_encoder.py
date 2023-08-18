import math
from typing import Optional

import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention

# Convert for Attention with Bidirectional Biases
def get_slopes(n_heads: int):
    n = 2 ** math.floor(math.log2(n_heads))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < n_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        # Concatenate the slopes with the remaining slopes.
        m = torch.cat([m, m_hat])

    return m


@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """

    # Get slopes $m$ for each head
    m = get_slopes(n_heads).to(mask.device)

    # Calculate distances $[0, 1, \dots, N]$
    # Here we calculate the distances using the mask.
    #
    # Since it's causal mask we can just use $[0, 1, \dots, N]$ too.
    # `distance = torch.arange(mask.shape[1], dtype=torch.long, device=mask.device)[None, :]`
    distance = mask.cumsum(dim=-1)

    # Multiply them pair-wise to get the AliBi bias matrix
    return distance[:, :, None] * m[None, None, :]


class AlibiMultiHeadAttention(MultiHeadAttention):
    """
    ## Attention with Linear Biases (ALiBi)

    We override [Multi-Head Attention](../mha.html).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)

        # To cache AliBi the biases
        self.alibi_biases = None

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # ALiBi only works with causal masks.
        assert mask is not None
        assert mask.shape[0] == mask.shape[1] and mask.shape[2] == 1

        # `query`, `key` and `value` have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        # Add head dimension to mask and check its shape.
        mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Create AliBi biases if it's not cached
        if self.alibi_biases is None or self.alibi_biases.shape[1] < seq_len:
            # `mask` has shape [seq_len, seq_len, 1, 1]
            self.alibi_biases = get_alibi_biases(scores.shape[-1], mask[:, :, 0, 0])

        # Add AliBi biases to attention scores.
        # ALiBi biases has shape `[seq_len, seq_len, n_heads]`
        # and `scores` has shape `[seq_len, seq_len, batch_size, n_heads]`
        scores += self.alibi_biases[:seq_len, :seq_len, None, :]

        # Apply mask
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)


def _test_alibi():
    """
    Simple test function to see the slopes.
    """
    inspect(get_slopes(12).tolist(), _n=-1)
    from labml_nn.transformers.utils import subsequent_mask

    mask = subsequent_mask(8)[:, :, 0]
    inspect(mask)

    inspect(get_alibi_biases(12, mask)[:, :, 3], _n=-1)


if __name__ == '__main__':
    _test_alibi()
