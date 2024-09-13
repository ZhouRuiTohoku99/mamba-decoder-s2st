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
)
from speechbrain.nnet.containers import ModuleList
from mamba_ssm import Mamba
logger = logging.getLogger(__name__)


class MambaS2ST():

    def __init__(
        self,
        d_model=512,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
        mamba_config = None,
        mt_src_vocab: int = 0,
    ):
        super().__init__()

        self.custom_mt_src_module = ModuleList(
            NormalizedEmbedding(d_model, mt_src_vocab)
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

        # reset parameters using xavier_normal_
        self._init_params()

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

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks_for_mt(src, tgt, pad_idx=pad_idx)
        
        tgt = self.custom_tgt_module(tgt)

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)

        decoder_out, _, multihead = self.decoder(
            tgt=tgt,
            memory=src,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return decoder_out

    def forward_s2st_decoder(self, src, tgt, tgt_lens = None):
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

        src_key_padding_mask = None
        tgt_key_padding_mask = None
        if tgt_lens is not None:
            abs_len = torch.round(tgt_lens * tgt.shape[1])
            tgt_key_padding_mask = ~length_to_mask(abs_len).bool()
        src_mask = None
        tgt_mask = None

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)

        decoder_out, _, multihead = self.decoder(
            tgt=tgt,
            memory=src,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return decoder_out


    def make_masks_for_mt(self, src, tgt, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder (required).
        tgt : torch.Tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).

        Returns
        -------
        src_key_padding_mask : torch.Tensor
            Timesteps to mask due to padding
        tgt_key_padding_mask : torch.Tensor
            Timesteps to mask due to padding
        src_mask : torch.Tensor
            Timesteps to mask for causality
        tgt_mask : torch.Tensor
            Timesteps to mask for causality
        """
        src_key_padding_mask = None
        if self.training:
            src_key_padding_mask = get_key_padding_mask(src, pad_idx=pad_idx)
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)

        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask
    
    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()

        tgt = self.custom_tgt_module(tgt)
        tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
        pos_embs_target = None
        pos_embs_encoder = None

   
        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[-1]
    
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
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt : torch.Tensor
            The sequence to the decoder layer (required).
        memory : torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask : torch.Tensor
            The mask for the tgt sequence (optional).
        memory_mask : torch.Tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask : torch.Tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask : torch.Tensor
            The mask for the memory keys per batch (optional).
        pos_embs_tgt : torch.Tensor
            The positional embeddings for the target (optional).
        pos_embs_src : torch.Tensor
            The positional embeddings for the source (optional).
        """
        output = tgt
        for dec_layer in self.layers:
            output, _, _ = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
        output = self.norm(output)

        return output, [None], [None]
