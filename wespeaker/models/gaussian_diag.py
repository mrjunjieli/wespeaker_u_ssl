"""NoAdapter MHFA with independent log-precision per head and feature."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GaussianDiag(nn.Module):
    def __init__(self, feat_dim=1024, layer_num=25, head_num=64,
                 embed_dim=256, compression_dim=128,
                 pooling_head_chunk_size=8, pooling_checkpoint=True):
        super().__init__()
        if pooling_head_chunk_size < 1:
            raise ValueError('pooling_head_chunk_size must be positive')
        self.layer_num = layer_num
        self.head_num = head_num
        self.cmp_dim = compression_dim
        self.pooling_head_chunk_size = pooling_head_chunk_size
        self.pooling_checkpoint = pooling_checkpoint
        self.weights_k = nn.Parameter(torch.ones(layer_num))
        self.weights_v = nn.Parameter(torch.ones(layer_num))
        self.cmp_linear_k = nn.Linear(feat_dim, compression_dim)
        self.cmp_linear_v = nn.Linear(feat_dim, compression_dim)
        self.norm_k = nn.LayerNorm(compression_dim)
        self.norm_v = nn.LayerNorm(compression_dim)
        self.att_head = nn.Linear(compression_dim, head_num * compression_dim)
        self.prior_mean = nn.Parameter(torch.zeros(1, head_num, compression_dim))
        self.prior_logprec = nn.Parameter(torch.zeros(1, head_num, compression_dim))
        self.bn_embed = nn.BatchNorm1d(head_num * compression_dim)
        self.pooling_fc = nn.Linear(head_num * compression_dim, embed_dim)

    def _project_and_pool(self, key, value, weight, bias, prior_mean, prior_logprec):
        with torch.autocast(device_type=key.device.type, enabled=False):
            logits = F.linear(key.float(), weight.float(), bias.float())
            logits = logits.reshape(key.shape[0], key.shape[1], -1, self.cmp_dim)
            logits = logits.clamp(-15.0, 15.0)
            prior_logprec = prior_logprec.float().clamp(-15.0, 15.0)
            log_normalizer = torch.logaddexp(
                torch.logsumexp(logits, dim=1), prior_logprec)
            weights = torch.exp(logits - log_normalizer.unsqueeze(1))
            mean = torch.einsum('bthd,btd->bhd', weights, value.float())
            mean = mean + torch.exp(prior_logprec - log_normalizer) * prior_mean.float()
            return mean, torch.exp(-log_normalizer)

    def _pool_from_key(self, key, value):
        means, variances = [], []
        for start in range(0, self.head_num, self.pooling_head_chunk_size):
            stop = min(start + self.pooling_head_chunk_size, self.head_num)
            inputs = (
                key, value,
                self.att_head.weight[start * self.cmp_dim:stop * self.cmp_dim],
                self.att_head.bias[start * self.cmp_dim:stop * self.cmp_dim],
                self.prior_mean[:, start:stop], self.prior_logprec[:, start:stop])
            if self.pooling_checkpoint and self.training and torch.is_grad_enabled():
                mean, variance = checkpoint(
                    self._project_and_pool, *inputs, use_reentrant=False)
            else:
                mean, variance = self._project_and_pool(*inputs)
            means.append(mean)
            variances.append(variance)
        return torch.cat(means, dim=1), torch.cat(variances, dim=1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            if len(x) != self.layer_num:
                raise ValueError('Unexpected number of frontend hidden states')
            x = torch.stack(x, dim=2)
        if x.ndim != 4 or x.shape[2] != self.layer_num:
            raise ValueError('Expected [batch, time, layer_num, input_dim]')
        key = (x * self.weights_k.softmax(dim=0)[None, None, :, None]).sum(dim=2)
        value = (x * self.weights_v.softmax(dim=0)[None, None, :, None]).sum(dim=2)
        key = self.norm_k(self.cmp_linear_k(key))
        value = self.norm_v(self.cmp_linear_v(value))
        mean, variance = self._pool_from_key(key, value)
        embedding = self.pooling_fc(self.bn_embed(mean.flatten(1)))
        with torch.autocast(device_type=value.device.type, enabled=False):
            variance = variance.flatten(1) * self.bn_embed.weight.float().square() / (
                self.bn_embed.running_var.float() + self.bn_embed.eps)
            embedding_variance = F.linear(
                variance, self.pooling_fc.weight.float().square())
        if self.training and hasattr(self, 'projection'):
            embedding = embedding + 0.0 * self.projection.weight.sum()
        return value, embedding_variance, embedding
