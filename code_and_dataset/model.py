import sys

import torch
import torch.nn as nn

from config_parser import Config

###################################################################
#
# Weight initial setup
#


def weight_initial(model, config):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.uniform_(param, a=-0.01, b=0.01)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            nn.init.constant_(m.bias.data, 0.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###################################################################
#
# Util class, same pad conv2d
#


class Conv1dSame(nn.Module):
    """Same padding conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, depth_wise=False, batch_norm=True):
        super(Conv1dSame, self).__init__()
        if depth_wise:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                  kernel_size, groups=in_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pad = nn.ReflectionPad1d((0, kernel_size - 1))
        self.batch_norm = batch_norm
        self.act = nn.LeakyReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, input_tensor):
        out = self.conv(self.pad(input_tensor))
        out = self.act(out)
        if self.batch_norm:
            out = self.bn(out)
        return out


###################################################################
#
# HLA lstm
#

class LSTM_Encoder(nn.Module):
    def __init__(self, device, input_channels, hidden_size, batch_size, layers, bias=False):
        super(LSTM_Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_channels,
            hidden_size,
            num_layers=layers,
            bias=bias,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        self.layers = layers
        self.num_directions = 2
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_states = None
        self.device = device
        self.zeros = torch.zeros((1)).to(device)
        self.act = nn.LeakyReLU()
        self.instNorm = nn.InstanceNorm1d(hidden_size * 2)

    def init_hidden_state(self):
        return (
            torch.zeros(self.layers * self.num_directions,
                        self.batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.layers * self.num_directions,
                        self.batch_size, self.hidden_size, device=self.device),
        )

    def forward(self, seq_input, seq_mask, seq_length):
        """
        int: [Batch, seq_len, encoding_dim], [Batch, max_len], [Batch,]
        out shape: [batch, seq_len, 2*hidden_size], [batch, 2*hidden_size]
        """
        self.hidden_states = self.init_hidden_state()
        out, self.hidden_states = self.lstm(seq_input, self.hidden_states)
        out = self.act(out)
        out_mask = seq_mask.view(seq_mask.size(0), seq_mask.size(1), 1).expand(
            seq_mask.size(0), seq_mask.size(1), out.size(-1))
        out_masked = torch.where(out_mask != 0.0, out, self.zeros)
        sum_hidden = torch.sum(out_masked, 1)

        return out_masked, sum_hidden

###################################################################
#
# Attention
#


class PepAttention(nn.Module):
    """A FC moduel to combine all encoder's information
    Use last hidden state to generate attention weight.
    Input:  [Batch, max_len, 2*hidden_size], [Batch, 2*hidden_size], [Batch, max_len]
    Output: [Batch, 2*hidden_size]
    """

    def __init__(self, max_seq_length, hidden_size, device):
        super(PepAttention, self).__init__()
        self.fc = nn.Linear(
            hidden_size * 2,
            max_seq_length,
        )
        self.sm = nn.Softmax(dim=1)
        self.zeros = torch.zeros((1)).to(device)
        self.instNorm = nn.InstanceNorm1d(hidden_size * 2)

    def forward(self, lstm_out, a_hidden, b_hidden, pep_hidden, seq_mask):
        lstm_hidden_state = a_hidden + b_hidden + pep_hidden
        lstm_hidden_state = self.instNorm(lstm_hidden_state.view(
            lstm_hidden_state.size(0), 1, lstm_hidden_state.size(1)))
        attn_weight = self.fc(lstm_hidden_state.view(
            lstm_hidden_state.size(0), -1))
        attn_weight = torch.where(seq_mask != 0.0, attn_weight, self.zeros)
        attn_weight = self.sm(attn_weight)
        attn_weight = torch.reshape(attn_weight, (attn_weight.size(
            0), 1, attn_weight.size(1)))
        out = torch.bmm(attn_weight, lstm_out)
        return out.view(out.size(0), -1), attn_weight.view(attn_weight.size(0), -1)


class Attention(nn.Module):
    """A FC moduel to combine all encoder's information
    Use last hidden state to generate attention weight.
    Input:  [Batch, max_len, 2*hidden_size], [Batch, 2*hidden_size], [Batch, max_len]
    Output: [Batch, 2*hidden_size]
    """

    def __init__(self, max_seq_length, hidden_size, device):
        super(Attention, self).__init__()
        self.fc = nn.Linear(
            hidden_size * 2,
            max_seq_length,
        )
        self.sm = nn.Softmax(dim=1)
        self.zeros = torch.zeros((1)).to(device)
        self.instNorm = nn.InstanceNorm1d(hidden_size * 2)

    def forward(self, lstm_out, lstm_hidden_state, seq_mask):
        lstm_hidden_state = self.instNorm(lstm_hidden_state.view(
            lstm_hidden_state.size(0), 1, lstm_hidden_state.size(1)))
        attn_weight = self.fc(lstm_hidden_state.view(
            lstm_hidden_state.size(0), -1))
        attn_weight = torch.where(seq_mask != 0.0, attn_weight, self.zeros)
        attn_weight = self.sm(attn_weight)
        attn_weight = torch.reshape(attn_weight, (attn_weight.size(
            0), 1, attn_weight.size(1)))
        out = torch.bmm(attn_weight, lstm_out)
        return out.view(out.size(0), -1), attn_weight.view(attn_weight.size(0), -1)

###################################################################
#
# Context extractor
#


class Context_extractor(nn.Module):
    """Extract context vector from 3 attention sources
    Input: List of attention tensor [Batch, 1, 2*hidden_size]
    Output: flattend vector
    """

    def __init__(self, hidden_size):
        super(Context_extractor, self).__init__()
        self.net = nn.Sequential(
            Conv1dSame(3, 64, 3),
            nn.MaxPool1d(2),

            Conv1dSame(64, 128, 3),
            nn.MaxPool1d(2),

            Conv1dSame(128, 256, 3),
            nn.MaxPool1d(2),

            Conv1dSame(256, 512, 3),
            nn.MaxPool1d(2),

            Conv1dSame(512, 1024, 3),
        )
        self.out_vector_dim = 12288

    def forward(self, list_tensors):
        out = torch.stack(list_tensors, dim=1)
        out = self.net(out)
        return out.view(out.size(0), -1)

###################################################################
#
# Predictor
#


class Predictor(nn.Module):
    """Predictor ic50  [0 - 1] from context vector
    """

    def __init__(self, input_size):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(50, 1)
        )
        self.out_act = nn.Sigmoid()

    def forward(self, context_vector):
        out = self.net(context_vector)
        return self.out_act(out)

###################################################################
#
# Binding core predictor
#


class BindingCorePredictor(nn.Module):
    """Binding core predictor
    """

    def __init__(self, hidden_size, max_seq_length, device):
        super(BindingCorePredictor, self).__init__()
        self.fc = nn.Linear(
            hidden_size * 2,
            max_seq_length,
        )
        self.zeros = torch.zeros((1)).to(device)
        self.sm = nn.Softmax(dim=1)
        self.instNorm = nn.InstanceNorm1d(hidden_size * 2)

    def forward(self, lstm_sum_hidden_a, lstm_sum_hidden_b, lstm_sum_hidden_pep, seq_mask):
        """
        Input: [batch, 2*hidden_size], [batch, 2*hidden_size], [batch, 2*hidden_size], [Batch, max_seq_len]
        Output: [Batch, max_seq_len]
        """
        attn_weight = lstm_sum_hidden_a + lstm_sum_hidden_b + lstm_sum_hidden_pep
        attn_weight = self.instNorm(attn_weight.view(
            attn_weight.size(0), 1, attn_weight.size(1)))
        attn_weight = self.fc(attn_weight.view(attn_weight.size(0), -1))
        attn_weight = torch.where(seq_mask != 0.0, attn_weight, self.zeros)
        attn_weight = self.sm(attn_weight)
        attn_weight = torch.reshape(attn_weight, (attn_weight.size(
            0), 1, attn_weight.size(1)))
        return attn_weight

###################################################################
#
# Model
#


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder_hla_a = LSTM_Encoder(
            config.device,
            config.seq_encode_dim,
            config.model_config['LSTM']['hidden_size'],
            config.batch_size,
            config.model_config['LSTM']['num_layers'],
        )
        self.attention_hla_a = Attention(
            config.max_len_hla_A,
            config.model_config['LSTM']['hidden_size'],
            config.device,
        )

        self.encoder_hla_b = LSTM_Encoder(
            config.device,
            config.seq_encode_dim,
            config.model_config['LSTM']['hidden_size'],
            config.batch_size,
            config.model_config['LSTM']['num_layers'],
        )
        self.attention_hla_b = Attention(
            config.max_len_hla_B,
            config.model_config['LSTM']['hidden_size'],
            config.device,
        )

        self.encoder_peptide = LSTM_Encoder(
            config.device,
            config.seq_encode_dim,
            config.model_config['LSTM']['hidden_size'],
            config.batch_size,
            config.model_config['LSTM']['num_layers'],
        )
        self.attention_pep = PepAttention(
            config.max_len_pep,
            config.model_config['LSTM']['hidden_size'],
            config.device,
        )

        self.binding_core_predictor = BindingCorePredictor(
            config.model_config['LSTM']['hidden_size'],
            config.max_len_pep,
            config.device,
        )

        self.context_extractor = Context_extractor(
            config.model_config['LSTM']['hidden_size'])

        self.predictor = Predictor(self.context_extractor.out_vector_dim)

    def forward(
        self, hla_a_seqs, hla_a_mask, hla_a_length,
        hla_b_seqs, hla_b_mask, hla_b_length,
        peptides, pep_mask, pep_length
    ):
        hla_a_out, hla_a_hidden = self.encoder_hla_a(
            hla_a_seqs, hla_a_mask, hla_a_length)
        hla_a_out, _ = self.attention_hla_a(
            hla_a_out, hla_a_hidden, hla_a_mask)

        hla_b_out, hla_b_hidden = self.encoder_hla_b(
            hla_b_seqs, hla_b_mask, hla_b_length)
        hla_b_out, _ = self.attention_hla_b(
            hla_b_out, hla_b_hidden, hla_b_mask)

        pep_out, pep_hidden = self.encoder_peptide(
            peptides, pep_mask, pep_length)
        pep_out, binding_core_weight = self.attention_pep(
            pep_out, hla_a_hidden, hla_b_hidden, pep_hidden, pep_mask)
        context = self.context_extractor([hla_a_out, hla_b_out, pep_out])
        ic50 = self.predictor(context)

        return ic50, binding_core_weight
