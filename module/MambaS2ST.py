"""Transformer for ST in the SpeechBrain style.

Authors
* YAO FEI, CHENG 2021
"""

import logging
import speechbrain as sb
import torch  # noqa 42
from torch import nn
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.Transformer import (
    NormalizedEmbedding,
    get_key_padding_mask,
    get_lookahead_mask,
    PositionalEncoding,
)
from speechbrain.nnet.containers import ModuleList
from mamba_ssm import Mamba
logger = logging.getLogger(__name__)


class MambaS2ST(nn.Module):

    def __init__(
        self,
        d_model=512,
        tgt_vocab = 103,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
        mamba_config = None,
        max_length = 2500,
    ):
        super().__init__()

        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )
        
        self.decoder = MambaDecoder(
                num_layers=num_decoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                activation=activation,
                dropout=dropout,
                normalize_before=normalize_before,
                mamba_config=mamba_config
            )
        self.positional_encoding = PositionalEncoding(d_model, max_length)

    def forward_mt_decoder_only(self, src, tgt, pad_idx=0):
        """This method implements a forward step for mt task using a wav2vec encoder
        (same than above, but without the encoder stack)

        Arguments
        ----------
        src (transcription): torch.Tensor
            output features from the w2v2 encoder
        tgt (translation): torch.Tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        
        tgt = self.custom_tgt_module(tgt)

        tgt = tgt + self.positional_encoding(tgt)

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=src,
        )

        return decoder_out

    
    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):

        tgt = self.custom_tgt_module(tgt)
        tgt = tgt + self.positional_encoding(tgt)  # add the encodings here

   
        prediction, _, _ = self.decoder(
            tgt,
            encoder_out,
        )
        return prediction, _
    
class MambaDecoderLayer(nn.Module):
    """This class implements the Mamba decoder layer.
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        activation=nn.ReLU,
        dropout=0.0,
        normalize_before=False,
        mamba_config=None
    ):
        super().__init__()

        assert mamba_config != None

        bidirectional = mamba_config.pop('bidirectional')

        self.self_mamba = Mamba(
            d_model=d_model,
            **mamba_config
        )

        self.cross_mamba = Mamba(
            d_model=d_model,
            **mamba_config
        )

        mamba_config['bidirectional'] = bidirectional

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm3 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        memory,
    ):
        """
        Arguments
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: torch.Tensor
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask: torch.Tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask: torch.Tensor
            The mask for the memory keys per batch (optional).
        pos_embs_tgt: torch.Tensor
            The positional embeddings for the target (optional).
        pos_embs_src: torch.Tensor
            The positional embeddings for the source (optional).
        """
        if self.normalize_before:
            tgt1 = self.norm1(tgt)
        else:
            tgt1 = tgt

        # Mamba over the target sequence
        tgt2 = self.self_mamba(tgt1)

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # Mamba over key=value + query
        # and only take the last len(query) tokens
        tgt2 = self.cross_mamba(torch.cat([memory, tgt1], dim=1))[:, -tgt1.shape[1]:]
        
        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, None, None


class MambaDecoder(nn.Module):
    """This class implements the Mamba decoder.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        activation=nn.ReLU,
        dropout=0.0,
        normalize_before=False,
        mamba_config=None
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                MambaDecoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    activation=activation,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    mamba_config=mamba_config
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
    ):
        """
        Arguments
        ----------
        tgt : torch.Tensor
            The sequence to the decoder layer (required).
        memory : torch.Tensor
            The sequence from the last layer of the encoder (required).
        """
        output = tgt
        for dec_layer in self.layers:
            output, _, _ = dec_layer(
                output,
                memory,
            )
        output = self.norm(output)

        return output, [None], [None]
