"""
VQA models.

BaselineNet featurizes the image and the question into 1-d vectors.
Then concatenates the two representations and feeds to a linear layer.
The backbones are frozen.

TransformerNet featurizes image-language into sequences,
then applies iterative self-/cross-attention on them.
Backbones are again frozen.
"""

import math

import torch
from torch import nn
from torchvision.models.resnet import resnet18
from transformers import RobertaModel, RobertaTokenizerFast


class BaselineNet(nn.Module):
    """Simple baseline for VQA."""

    def __init__(self, n_answers=5217):
        """Initialize layers given the number of answers."""
        super().__init__()
        # Text encoder
        t_type = "roberta-base"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Visual encoder
        r18 = resnet18(pretrained=True)
        self.vis_encoder = nn.Sequential(*(list(r18.children())[:-2]))
        for param in self.vis_encoder.parameters():
            param.requires_grad = False

        # Classifier
        self.classifier = nn.Linear(
            self.text_encoder.config.hidden_size + 512,
            TODO
        )

    def forward(self, image, question):
        """Forward pass, image (B, 3, 224, 224), qs list of str."""
        return TODO

    @torch.no_grad()
    def compute_text_feats(self, text):
        """Convert list of str to feature tensors."""
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="longest", return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)
        encoded_text = self.text_encoder(**tokenized)
        return encoded_text.pooler_output

    @torch.no_grad()
    def compute_vis_feats(self, image):
        """Convert image tensors to feature tensors."""
        return TODO  # feed to vis_encoder and then mean pool on spatial dims

    def train(self, mode=True):
        """Override train to set backbones in eval mode."""
        nn.Module.train(self, mode=mode)
        self.vis_encoder.eval()
        self.text_encoder.eval()


class TransformerNet(BaselineNet):
    """Simple transformer-based model for VQA."""

    def __init__(self, n_answers=5217):
        """Initialize layers given the number of answers."""
        super().__init__()
        # Text/visual encoders are inhereted from BaselineNet
        # Positional embeddings
        self.pos_embed = PositionEmbeddingSine(128, normalize=True)

        # Project to 256
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 256)
        self.vis_proj = nn.Linear(512, 256)

        # Cross-attention
        self.ca = nn.ModuleList([CrossAttentionLayer() for _ in range(3)])

        # Classifier
        self.classifier = nn.Linear(256 + 256, n_answers)

    def forward(self, image, question):
        """Forward pass, image (B, 3, 224, 224), qs list of str."""
        # Per-modality encoders
        text_feats, text_mask = self.compute_text_feats(question)
        text_feats = self.text_proj(text_feats)
        vis_feats, vis_pos = self.compute_vis_feats(image)
        vis_feats = self.vis_proj(vis_feats)
        vis_pos = vis_pos.to(vis_feats.device)

        # Cross-encoder
        for layer in self.ca:
            vis_feats, text_feats = layer(
                vis_feats, None,
                text_feats, text_mask,
                seq1_pos=vis_pos
            )

        # Classifier
        return self.classifier(torch.cat((
            (text_feats * (~text_mask)[..., None].float()).sum(1)
            / (~text_mask)[..., None].float().sum(1),
            vis_feats.mean(1)
        ), 1))

    @torch.no_grad()
    def compute_text_feats(self, text):
        """Convert list of str to feature tensors."""
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="longest", return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)
        encoded_text = self.text_encoder(**tokenized)
        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        return encoded_text.last_hidden_state, text_attention_mask

    @torch.no_grad()
    def compute_vis_feats(self, image):
        """Convert image tensors to feature tensors."""
        encoded_img = self.vis_encoder(image)
        B, F, D, _ = encoded_img.shape
        return (
            encoded_img.reshape(B, F, D * D).transpose(1, 2),
            self.pos_embed(D).reshape(1, -1, D * D).transpose(1, 2)
        )


class CrossAttentionLayer(nn.Module):
    """Self-/Cross-attention between two sequences."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        # Self-attention for seq1
        self.sa1 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )  # use batch_first=True everywhere!
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        # Self-attention for seq2
        self.sa2 = TODO
        self.dropout_2 = TODO
        self.norm_2 = TODO

        # Cross attention from seq1 to seq2
        self.cross_12 = TODO
        self.dropout_12 = TODO
        self.norm_12 = TODO

        # FFN for seq1
        self.ffn_12 = nn.Sequential(TODO)  # hidden dim is 1024
        self.norm_122 = TODO

        # Cross attention from seq2 to seq1
        self.cross_21 = TODO
        self.dropout_21 = TODO
        self.norm_21 = TODO

        # FFN for seq2
        self.ffn_21 = nn.Sequential(TODO)  # hidden dim is 1024
        self.norm_212 = TODO

    def forward(self, seq1, seq1_key_padding_mask, seq2,
                seq2_key_padding_mask,
                seq1_pos=None, seq2_pos=None):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        # Self-attention for seq1
        q1 = k1 = v1 = seq1
        if seq1_pos is not None:
            q1 = q1 + seq1_pos
            k1 = k1 + seq1_pos
        seq1b = self.sa1(
            query=q1,
            key=k1,
            value=v1,
            attn_mask=None,
            key_padding_mask=seq1_key_padding_mask  # (B, S1)
        )[0]
        seq1 = self.norm_1(seq1 + self.dropout_1(seq1b))

        # Self-attention for seq2
        q2 = k2 = v2 = seq2
        if seq2_pos is not None:
            q2 = q2 + seq2_pos
            k2 = k2 + seq2_pos
        TODO

        # Create key, query, value for seq1, seq2
        q1 = k1 = v1 = seq1
        q2 = k2 = v2 = seq2
        if seq1_pos is not None:
            q1 = q1 + seq1_pos
            k1 = k1 + seq1_pos
        if seq2_pos is not None:
            q2 = q2 + seq2_pos
            k2 = k2 + seq2_pos

        # Cross-attention from seq1 to seq2 and FFN
        TODO

        # Cross-attention from seq2 to seq1 and FFN
        TODO

        return seq1, seq2


class PositionEmbeddingSine(nn.Module):
    """
    2-D version of position embeddings,
    similar to the "Attention is all you need" paper.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, spat_dim):
        mask = torch.zeros(1, spat_dim, spat_dim).bool()
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
