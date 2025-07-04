import torch
import torch.nn as nn
from models.transformer import TransformerBlock

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.mag_to_phase_attn = TransformerBlock(d_model=dim, n_heads=n_heads)
        self.phase_to_mag_attn = TransformerBlock(d_model=dim, n_heads=n_heads)

    def forward(self, mag_feats, phase_feats):
        b, c, t, f = mag_feats.size()

        # Приводим к формату [batch, sequence_length, feature_dim]
        mag_feats_flat = mag_feats.permute(0, 2, 3, 1).contiguous().view(b, t * f, c)
        phase_feats_flat = phase_feats.permute(0, 2, 3, 1).contiguous().view(b, t * f, c)

        updated_mag_feats = mag_feats_flat + self.phase_to_mag_attn(mag_feats_flat, phase_feats_flat, phase_feats_flat)
        updated_phase_feats = phase_feats_flat + self.mag_to_phase_attn(phase_feats_flat, mag_feats_flat,
                                                                        mag_feats_flat)

        # Возвращаем исходную размерность
        updated_mag_feats = updated_mag_feats.view(b, t, f, c).permute(0, 3, 1, 2)
        updated_phase_feats = updated_phase_feats.view(b, t, f, c).permute(0, 3, 1, 2)

        return updated_mag_feats, updated_phase_feats
