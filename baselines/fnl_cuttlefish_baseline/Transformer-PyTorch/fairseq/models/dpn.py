import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import math

from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.modules import LearnedPositionalEmbedding, LayerNormalization, SinusoidalPositionalEmbedding, LinearizedConvolution

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model, register_model_architecture

@register_model('dpn')
class DPModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--hidden_size', type=int, metavar='N',
                            help='The hidden size')
        parser.add_argument('--filter_size', type=int, metavar='N',
                            help='The filter size')
        parser.add_argument('--num_layers', type=int, metavar='N',
                            help='The number of hidden layer')
        parser.add_argument('--num_head', type=int, metavar='N',
                            help='The number of heads')
        parser.add_argument('--position', type=str, metavar='P',
                            default='learned', choices=['learned', 'timing', 'nopos'],
                            help='The method of learning position')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings')

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        if not hasattr(args, 'share_input_output_embed'):
            args.share_input_output_embed = False
        encoder = DPEncoder(
            src_dict,
            embed_dim=args.hidden_size,
            hidden_size=args.hidden_size,
            filter_size=args.filter_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pos=args.position,
        )
        decoder = DPDecoder(
            dst_dict,
            embed_dim=args.hidden_size,
            hidden_size=args.hidden_size,
            filter_size=args.filter_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pos=args.position,
            share_embed=args.share_input_output_embed,
        )
        return DPModel(encoder, decoder)


class AttnPathEncoder(nn.Module):
    def __init__(self,
                 num_layers=2, num_heads=8, 
                 filter_size=256, hidden_size=256,
                 dropout=0.1, attention_dropout=0.1, relu_dropout=0.1):
        super(AttnPathEncoder, self).__init__()
        self.layers = num_layers
        self.self_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttention(hidden_size,
                                                                 hidden_size,
                                                                 hidden_size,
                                                                 num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(hidden_size, filter_size, relu_dropout))
            self.norm1_blocks.append(LayerNormalization(hidden_size))
            self.norm2_blocks.append(LayerNormalization(hidden_size))
        self.out_norm = LayerNormalization(hidden_size)

    def forward(self, x):
        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            y = self_attention(norm1(x), None, encoder_self_attention_bias)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
        x = self.out_norm(x)
        return x


class CNNPathEncoder(nn.Module):
    def __init__(self,
                 num_layers=4, hidden_size=256,
                 dropout=0.1, in_embed=256, out_embed=256):
        super(CNNPathEncoder, self).__init__()
        
        self.layers = num_layers
        self.dropout = dropout
        self.fc1 = Linear(in_embed, filter_size, dropout=dropout)
        self.convolutions = nn.ModuleList()

        kernel_size = 3
        
        for i in range(num_layers):
            padding = 1
            self.convolutions.append(
                ConvTBC(hidden_size, hidden_size * 2, kernel_size,
                        dropout=dropout, padding=padding)
            )
        self.fc2 = Linear(filter_size, out_embed)

    def forward(self, x):
        x = self.fc1(x)
        x = x.transpose(0, 1)
        for conv in zip(self.convolutions):
            residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=2)
            x = (x + residual) * math.sqrt(0.5)
        x = x.transpose(1, 0)
        x = self.fc2(x)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.layers))
        return x


class DPEncoder(FairseqEncoder):
    """Transformer encoder."""
    def __init__(self, dictionary, embed_dim=256, max_positions=1024, pos="learned",
                 num_layers=2, num_heads=8,
                 filter_size=256, hidden_size=256,
                 dropout=0.1, attention_dropout=0.1, relu_dropout=0.1,
                 convolutions=4):
        super().__init__(dictionary)
        assert pos == "learned" or pos == "timing" or pos == "nopos"

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.pos = pos

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if self.pos == "learned":
            self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx,
                                                       left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)
        if self.pos == "timing":
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, padding_idx,
                                                                 left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)

        self.layers = num_layers
        self.attnpath = AttnPathEncoder(self.layers, num_heads=num_heads,
                                        filter_size=filter_size, hidden_size=hidden_size,
                                        dropout=dropout, attention_dropout=attention_dropout,
                                        relu_dropout=relu_dropout)
        self.cnnpath = CNNPathEncoder(self.layers, hidden_size=hidden_size, dropout=dropout,
                                      in_embed=hidden_size, out_embed=hidden_size)


    def forward(self, src_tokens, src_lengths):
        # embed tokens plus positions
        input_to_padding = attention_bias_ignore_padding(src_tokens, self.dictionary.pad())
        encoder_self_attention_bias = encoder_attention_bias(input_to_padding)
        encoder_input = self.embed_tokens(src_tokens)
        if self.pos != "nopos":
            encoder_input += self.embed_positions(src_tokens)

        x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        
        attn_x = self.attnpath(x)
        cnn_x = self.cnnpath(x)

        return (attn_x, cnn_x)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.pos == "learned":
            return self.embed_positions.max_positions()
        else:
            return 1024


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))


class AttnPathDecoder(nn.Module):
    def __init__(self,
                 num_layers=2, num_heads=8,
                 filter_size=256, hidden_size=256,
                 dropout=0.1, attention_dropout=0.1, relu_dropout=0.1):
        super(AttnPathDecoder, self).__init__()
        self.layers = num_layers

        self.self_attention_blocks = nn.ModuleList()
        self.encdec_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()
        self.norm3_blocks = nn.ModuleList()
        self.gate_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttentionDecoder(hidden_size,
                                                                        hidden_size,
                                                                        hidden_size,
                                                                        num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(hidden_size, filter_size, relu_dropout))
            self.norm1_blocks.append(LayerNormalization(hidden_size))
            self.norm2_blocks.append(LayerNormalization(hidden_size))
            self.norm3_blocks.append(LayerNormalization(hidden_size))
            self.encdec_attention_blocks.append(MultiheadAttention(hidden_size,
                                                                   hidden_size,
                                                                   hidden_size,
                                                                   num_heads))
        self.out_norm = LayerNormalization(hidden_size)
    
    def forward(self, x, encoder_out, incremental_state=None):
        encoder_out_a, encoder_out_c = encoder_out

        avg_attn_scores = None
        num_attn_layers = len(self.encdec_attention_blocks)
        for self_attention, encdec_attention, ffn, norm1, norm2, norm3 in zip(self.self_attention_blocks,
                                                                              self.encdec_attention_blocks,
                                                                              self.ffn_blocks,
                                                                              self.norm1_blocks,
                                                                              self.norm2_blocks,
                                                                              self.norm3_blocks):
            y = self_attention(norm1(x), None, decoder_self_attention_bias, incremental_state)
            x = residual(x, y, self.dropout, self.training)
            
            y0, attn_scores = encdec_attention(norm2(x), encoder_out_a, None, True)
            y1, attn_scores = encdec_attention(norm2(x), encoder_out_c, None, True)
            attn_scores = attn_scores / self.layers
            if avg_attn_scores is None:
                avg_attn_scores = attn_scores
            else:
                avg_attn_scores.add_(attn_scores)
            
            y = y0 + y1
            x = residual(x, y, self.dropout, self.training)

            y = ffn(norm3(x))
            x = residual(x, y, self.dropout, self.training)
        x = self.out_embed(self.out_norm(x))
        return x, avg_attn_scores
        


class CNNPathDecoder(nn.Module):
    def __init__(self, 
                 num_layers=4, hidden_size=256,
                 dropout=0.1, embed_dim=256):
        super(CNNPathDecoder, self).__init__()

        self.layers  = num_layers
        self.dropout = dropout
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        kernel_size = 3

        for i in range(num_layers):
            pad = kernel_size - 1
            self.convolutions.append(
                LinearizedConv1d(hidden_size, hidden_size * 2, kernel_size,
                                 padding=(kernel_size - 1), dropout=dropout)
            )
            self.attention.append(AttentionLayer(hidden_size, embed_dim))

    def forward(self, x, encoder_out, incremental_state=None):
        encoder_out_a, encoder_out_c = encoder_out

        target_embedding = x
        x = self._transpose_if_training(x, incremental_state)

        for conv, attention in zip(self.convolutions, self.attention):
            residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)
                
            x0, attn_scores = attention(x, target_embedding, (encoder_out_a, encoder_out_a))
            x1, attn_scores = attention(x, target_embedding, (encoder_out_c, encoder_out_c))
            attn_scores = attn_scores / num_attn_layers

            x = self._transpose_if_training(x, incremental_state)

            x = (x + residual) * math.sqrt(0.5)
        return x, avg_attn_scores


class DPDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""
    def __init__(self, dictionary, embed_dim=256, max_positions=1024, pos="learned",
                 num_layers=2, num_heads=8,
                 filter_size=256, hidden_size=256,
                 dropout=0.1, attention_dropout=0.1, relu_dropout=0.1, share_embed=False,
                 convolutions=((256, 3),) * 8):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        assert pos == "learned" or pos == "timing" or pos == "nopos"

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.pos = pos

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if self.pos == "learned":
            self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx,
                                                       left_pad=LanguagePairDataset.LEFT_PAD_TARGET)
        if self.pos == "timing":
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, padding_idx,
                                                                 left_pad=LanguagePairDataset.LEFT_PAD_TARGET)

        self.layers = num_layers

        self.self_attention_blocks = nn.ModuleList()
        self.encdec_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()
        self.norm3_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttentionDecoder(hidden_size,
                                                                        hidden_size,
                                                                        hidden_size,
                                                                        num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(hidden_size, filter_size, relu_dropout))
            self.norm1_blocks.append(LayerNormalization(hidden_size))
            self.norm2_blocks.append(LayerNormalization(hidden_size))
            self.norm3_blocks.append(LayerNormalization(hidden_size))
            self.encdec_attention_blocks.append(MultiheadAttention(hidden_size,
                                                                   hidden_size,
                                                                   hidden_size,
                                                                   num_heads))
        self.out_norm = LayerNormalization(hidden_size)

        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        in_channels = hidden_size
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            pad = kernel_size - 1
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                LinearizedConv1d(in_channels, out_channels * 2, kernel_size,
                                padding=(kernel_size - 1), dropout=dropout)    
            )
            self.attention.append(AttentionLayer(out_channels, embed_dim))
            in_channels = out_channels
        if share_embed:
            assert out_embed_dim == embed_dim, \
                "Shared embed weights implies same dimensions " \
                " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
            self.out_embed = nn.Linear(hidden_size, num_embeddings)
            self.out_embed.weight = self.embed_tokens.weight
        else:
            self.out_embed = Linear(hidden_size, num_embeddings, dropout=dropout)

    def forward(self, input_tokens, encoder_out, incremental_state=None):
        # split and transpose encoder outputs
        
        encoder_out_1, encoder_out_2 = encoder_out

        input_to_padding = attention_bias_ignore_padding(input_tokens, self.dictionary.pad())
        decoder_self_attention_bias = encoder_attention_bias(input_to_padding)
        decoder_self_attention_bias += attention_bias_lower_triangle(input_tokens)
        # embed positions

        positions = self.embed_positions(input_tokens, incremental_state)
        if incremental_state is not None:
            input_tokens = input_tokens[:, -1:]
            decoder_self_attention_bias = decoder_self_attention_bias[:, -1:, :]

        # embed tokens and positions
        x = self.embed_tokens(input_tokens) + positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        z = x

        avg_attn_scores = None
        num_attn_layers = len(self.encdec_attention_blocks)
        for self_attention, encdec_attention, ffn, norm1, norm2, norm3 in zip(self.self_attention_blocks,
                                                                              self.encdec_attention_blocks,
                                                                              self.ffn_blocks,
                                                                              self.norm1_blocks,
                                                                              self.norm2_blocks,
                                                                              self.norm3_blocks):
            y = self_attention(norm1(x), None, decoder_self_attention_bias, incremental_state)
            x = residual(x, y, self.dropout, self.training)
            
            if incremental_state is not None:
                y, attn_scores = encdec_attention(norm2(x), encoder_out, None, True)
                attn_scores = attn_scores / self.layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)
            else:
                y = encdec_attention(norm2(x), encoder_out, None)
            x = residual(x, y, self.dropout, self.training)

            y = ffn(norm3(x))
            x = residual(x, y, self.dropout, self.training)
        x = self.out_embed(self.out_norm(x))

        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            res = z if proj is None else proj(z)

            z = F.dropout(z, p=self.dropout, training=self.training)
            z = conv(z, incremental_state)
            z = F.glu(z, dim=2)

        return x, avg_attn_scores

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if state_dict.get('decoder.version', torch.Tensor([1]))[0] < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x


class MultiheadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, 
                 key_depth, value_depth, output_depth,
                 num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        self._query = Linear(key_depth, key_depth, bias=False)
        self._key = Linear(key_depth, key_depth, bias=False)
        self._value = Linear(value_depth, value_depth, bias=False)
        self.output_perform = Linear(value_depth, output_depth, bias=False)

        self.num_heads = num_heads
        self.key_depth_per_head = key_depth // num_heads
        self.dropout = dropout
        
    def forward(self, query_antecedent, memory_antecedent, bias, to_weights=False):
        if memory_antecedent is None:
            memory_antecedent = query_antecedent
        q = self._query(query_antecedent)
        k = self._key(memory_antecedent)
        v = self._value(memory_antecedent)
        q *= self.key_depth_per_head ** -0.5
        
        # split heads
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        x = []
        avg_attn_scores = None
        for i in range(self.num_heads):
            results = dot_product_attention(q[i], k[i], v[i],
                                            bias,
                                            self.dropout,
                                            to_weights)
            if to_weights:
                y, attn_scores = results
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)
            else:
                y = results
            x.append(y)
        x = combine_heads(x)
        x = self.output_perform(x)
        if to_weights:
            return x, avg_attn_scores / self.num_heads
        else:
            return x


class MultiheadAttentionDecoder(MultiheadAttention):
    """Multihead Attention for incremental eval"""
    def __init__(self,
                 key_depth, value_depth, output_depth,
                 num_heads, dropout=0.1):
        super(MultiheadAttentionDecoder, self).__init__(key_depth, value_depth, output_depth,
                                                        num_heads, dropout)

    def forward(self, query_antecedent, memory_antecedent, bias, incremental_state=None):
        if incremental_state is not None:
            return self.incremental_forward(query_antecedent, memory_antecedent, bias, incremental_state)
        else:
            return super().forward(query_antecedent, memory_antecedent, bias)
    
    def incremental_forward(self, query_antecedent, memory_antecedent, bias, incremental_state):
        if memory_antecedent is None:
            memory_antecedent = query_antecedent
        q = self._query(query_antecedent)
        k = self._key(memory_antecedent)
        v = self._value(memory_antecedent)
        q *= self.key_depth_per_head ** -0.5

        key_buffer = self._get_input_buffer(incremental_state, 'key_buffer')
        value_buffer = self._get_input_buffer(incremental_state, 'value_buffer')
        if key_buffer is None:
            key_buffer = k.clone()
            value_buffer = v.clone()
        else:
            key_buffer = torch.cat([key_buffer, k], 1)
            value_buffer = torch.cat([value_buffer, v], 1)

        self._set_input_buffer(incremental_state, key_buffer, 'key_buffer')
        self._set_input_buffer(incremental_state, value_buffer, 'value_buffer')

        k = key_buffer
        v = value_buffer
        
        # split heads
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        x = []
        for i in range(self.num_heads):
            x.append(dot_product_attention(q[i], k[i], v[i], bias, self.dropout))
        x = combine_heads(x)
        x = self.output_perform(x)
        return x

    def reorder_incremental_state(self, incremental_state, new_order):
        key_buffer = self._get_input_buffer(incremental_state, 'key_buffer')
        value_buffer = self._get_input_buffer(incremental_state, 'value_buffer')
        if key_buffer is not None:
            key_buffer.data = key_buffer.data.index_select(0, new_order)
            value_buffer.data = value_buffer.data.index_select(0, new_order)
            self._set_input_buffer(incremental_state, key_buffer, 'key_buffer')
            self._set_input_buffer(incremental_state, value_buffer, 'value_buffer')

    def _get_input_buffer(self, incremental_state, name):
        return utils.get_incremental_state(self, incremental_state, name)

    def _set_input_buffer(self, incremental_state, new_buffer, name):
        return utils.set_incremental_state(self, incremental_state, name, new_buffer)


class GatedNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(GatedNetwork, self).__init__()
        self.fc = Linear(hidden_size, hidden_size)

    def forward(self, x1, x2):
        gate = F.sigmoid(self.fc(x1 + x2))
        return gate * x1 + (1 - gate) * x2


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = Linear(hidden_size, filter_size, bias=False)
        self.fc2 = Linear(filter_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def residual(x, y, dropout, training):
    """Residual connection"""
    y = F.dropout(y, p=dropout, training=training)
    return x + y


def split_heads(x, num_heads):
    """split x into multi heads
    Args:
        x: [batch_size, length, depth]
    Returns:
        y: [[batch_size, length, depth / num_heads] x heads]
    """
    sz = x.size()
    # x -> [batch_size, length, heads, depth / num_heads]
    x = x.view(sz[0], sz[1], num_heads, sz[2] // num_heads)
    # [batch_size, length, 1, depth // num_heads] * 
    heads = torch.chunk(x, num_heads, 2)
    x = []
    for i in range(num_heads):
        x.append(torch.squeeze(heads[i], 2))
    return x


def combine_heads(x):
    """combine multi heads
    Args:
        x: [batch_size, length, depth / num_heads] x heads
    Returns:
        x: [batch_size, length, depth]
    """
    return torch.cat(x, 2)
    

def dot_product_attention(q, k, v, bias, dropout, to_weights=False):
    """dot product for query-key-value
    Args:
        q: query antecedent, [batch, length, depth]
        k: key antecedent,   [batch, length, depth]
        v: value antecedent, [batch, length, depth]
        bias: masked matrix
        dropout: dropout rate
        to_weights: whether to print weights
    """
    # [batch, length, depth] x [batch, depth, length] -> [batch, length, length]
    logits = torch.bmm(q, k.transpose(1, 2).contiguous())
    if bias is not None:
        logits += bias
    size = logits.size()
    weights = F.softmax(logits.view(size[0] * size[1], size[2]), dim=1)
    weights = weights.view(size)
    if to_weights:
        return torch.bmm(weights, v), weights
    else:
        return torch.bmm(weights, v)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def attention_bias_ignore_padding(src_tokens, padding_idx):
    """Calculate the padding mask based on which embedding are zero
    Args:
        src_tokens: [batch_size, length]
    Returns:
        bias: [batch_size, length]
    """
    return src_tokens.eq(padding_idx).unsqueeze(1)


def attention_bias_lower_triangle(input_tokens):
    batch_size, length = input_tokens.size()
    attention_bias_lower_triangle = torch.triu(input_tokens.data.new(length, length).fill_(1), diagonal=1)
    return Variable(attention_bias_lower_triangle.expand(batch_size, length, length).float()) * -1e9


def encoder_attention_bias(bias):
    batch_size, _, length = bias.size()
    return bias.expand(batch_size, length, length).float() * -1e9


def get_timing_signal_1d(position, num_timescales, min_timescale=1.0, max_timescale=1.0e4):
    """A pytorch implementation extended from tensor2tensor, just for comparsion
    """
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / 
        (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    Args:
        x: [batch_size, length, channels]
    """
    signal = get_timing_signal_1d(torch.arange(x.size(1)), 
                                  x.size(2) // 2,
                                  min_timescale,
                                  max_timescale)
    signal = signal.unsqueeze(0)
    signal = signal.type_as(x.data)
    x.data.add_(signal)
    return x


@register_model_architecture('dpn', 'dpn')
def base_architecture(args):
    args.hidden_size = getattr(args, 'hidden_size', 256)
    args.filter_size = getattr(args, 'filter_size', 1024)
    args.num_heads = getattr(args, 'num_heads', 4)
    args.num_layers = getattr(args, 'num_layers', 2)
    args.label_smoothing = getattr(args, 'label_smoothing', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
