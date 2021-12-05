import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        self.w_q = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding='same')
        self.w_k = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding='same')
        self.w_v = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding='same')
        self.attention_size = hidden_size // n_heads
        self.fc = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding='same')

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.attention_size).permute(0, 2, 1, 3)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.attention_size).permute(0, 2, 3, 1)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.attention_size).permute(0, 2, 1, 3)

        attn = nn.Softmax(dim=-1)(torch.matmul(q, k) / self.attention_size)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, self.n_heads * self.attention_size, -1)

        output = self.fc(output)

        return output


class FFTBlock(nn.Module):
    def __init__(self, hidden_size, n_head, conv_hidden_size):
        super(FFTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadAttention(hidden_size, n_head)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, conv_hidden_size, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(conv_hidden_size, hidden_size, kernel_size=3, padding='same')

    def forward(self, x):
        pre = self.norm1(x).permute(0, 2, 1)
        x = x.permute(0, 2, 1) + self.self_attn(pre, pre, pre)
        x = x.permute(0, 2, 1)

        pre = self.norm2(x).permute(0, 2, 1)
        tmp = self.conv1(pre)
        tmp = nn.ReLU()(tmp)
        tmp = self.conv2(tmp)
        x = tmp + x.permute(0, 2, 1)

        return x.permute(0, 2, 1)


# какой-то туториал про трансформеры на pytorch (в других курсах юзал, так что формально с себя копипастил уже))0)
# https://pytorch.org/tutorials/beginner/translation_transformer.html
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return token_embedding + self.pos_embedding[:token_embedding.size(0), :]


class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(PhonemeEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class DurationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DurationPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding='same'),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding='same'),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = layer(x)
        return x


class LengthRegulator(nn.Module):
    def __init__(self, input_size, hidden_size, alpha=1.0):
        super(LengthRegulator, self).__init__()
        self.dp = DurationPredictor(input_size, hidden_size)
        self.alpha = alpha

    def forward(self, x, true_durations=None):
        log_lengths = self.dp(x).squeeze(-1) * self.alpha
        lengths = log_lengths.detach().exp().cpu().int.numpy()
        res = []
        if self.training:
            lengths = true_durations.cpu().int.numpy()

        for i in range(x.shape[0]):
            res.append(torch.repeat_interleave(x[i], lengths[i], 0))
        res = pad_sequence(res, batch_first=True)
        return res.to('cuda'), log_lengths


class FastSpeech(nn.Module):
    def __init__(self, out_size, phoneme_vocab_size, hidden_size, n_head, fft_hidden_size=1536):
        super(FastSpeech, self).__init__()
        self.positional_encoding1 = PositionalEncoding(hidden_size)
        self.phoneme_embed = PhonemeEmbedding(phoneme_vocab_size, hidden_size)
        self.fft1 = nn.Sequential(
            *[FFTBlock(hidden_size, n_head, fft_hidden_size) for _ in range(6)]
        )
        self.len_reg = LengthRegulator(hidden_size, hidden_size)
        self.positional_encoding2 = PositionalEncoding(hidden_size)
        self.fft2 = nn.Sequential(
            *[FFTBlock(hidden_size, n_head, fft_hidden_size) for _ in range(6)]
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, tokens, aligned):
        x = self.positional_encoding1(self.phoneme_embed(tokens))
        x = self.fft1(x)
        x, lengths = self.len_reg(x, aligned)
        x = self.positional_encoding2(x)
        x = self.fft2(x)
        x = self.fc(x)
        return x.permute(0, 2, 1), lengths
