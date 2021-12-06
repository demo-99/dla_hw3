import PIL
import numpy as np
import torch

from dataclasses import dataclass

import torchaudio
from torchvision.transforms import ToTensor

from fastspeech import FastSpeech
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aligner import GraphemeAligner
from dataset import LJSpeechDataset, LJSpeechCollator
from featurizer import MelSpectrogram, MelSpectrogramConfig
from utils import plot_spectrogram_to_buf
from vocoder import Vocoder
from writer import WanDBWriter


@dataclass
class ModelConfig:
    out_size: int = 80
    phoneme_vocab_size: int = 51
    hidden_size: int = 384
    n_head: int = 2


NUM_EPOCHS = 15
BATCH_SIZE = 48
VALIDATION_TRANSCRIPTS = [
    'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
    'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
    'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space',
]


aligner = GraphemeAligner().to('cuda')
dataloader = DataLoader(LJSpeechDataset('.'), batch_size=BATCH_SIZE, collate_fn=LJSpeechCollator())
featurizer = MelSpectrogram(MelSpectrogramConfig())
vocoder = Vocoder().to('cuda:0').eval()
writer = WanDBWriter()

model = FastSpeech(
    ModelConfig.out_size,
    ModelConfig.phoneme_vocab_size,
    ModelConfig.hidden_size,
    ModelConfig.n_head
).to('cuda')

try:
    model.load_state_dict(torch.load('fastspeech_checkpoint'))
except:
    pass

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, betas=(.9, .98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, 1e-7)

tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

val_batch = tokenizer(VALIDATION_TRANSCRIPTS)[0].to('cuda')

loss_log = []
for e in range(NUM_EPOCHS):
    model.train()
    loss_iter = np.array([])
    for i, batch in tqdm(enumerate(dataloader)):
        melspec = featurizer(batch.waveform).to('cuda')
        melspec_length = melspec.size(-1) - (melspec == -11.5129251)[:, 0, :].sum(dim=-1)

        batch.durations = aligner(
            batch.waveform.to('cuda'),
            batch.waveform_length.to('cuda'),
            batch.transcript
        ).to('cuda') * melspec_length.unsqueeze(-1)

        optimizer.zero_grad()

        pred, lenghts_pred = model(batch.tokens.to('cuda'), batch.durations)
        min_len = min(batch.durations.size(-1), lenghts_pred.size(-1))
        length_loss = loss_fn(
            batch.durations[:, :min_len],
            lenghts_pred[:, :min_len].exp()
        )

        min_len = min(melspec.size(-1), pred.size(-1))
        melspec_loss = loss_fn(melspec[:, :, :min_len].to('cuda'), pred[:, :, :min_len])

        loss = length_loss + melspec_loss

        loss.backward()
        optimizer.step()

        length_loss = length_loss.detach().cpu().numpy()
        melspec_loss = melspec_loss.detach().cpu().numpy()
        loss_iter = np.append(loss_iter, length_loss + melspec_loss)

        if i % 10 == 9:
            writer.set_step(e * len(dataloader) + i)
            print('train loss: {}'.format(loss_iter[i-9:].mean()))
            writer.add_scalar('learning rate', scheduler.get_last_lr()[0])
            writer.add_scalar('train loss', loss_iter[i-9:].mean())
        scheduler.step()
    loss_iter = loss_iter.mean()
    loss_log.append(loss_iter.mean())
    with torch.no_grad():
        model.eval()
        generated_waves = vocoder.inference(model(val_batch, None)[0]).cpu()

        for name, p in model.named_parameters():
            writer.add_histogram(name, p, bins="auto")

        for audio, t in zip(generated_waves, VALIDATION_TRANSCRIPTS):
            image = PIL.Image.open(plot_spectrogram_to_buf(audio))
            writer.add_image("Waveform for '{}'".format(t), ToTensor()(image))
            writer.add_audio("Audio for '{}'".format(t), audio, MelSpectrogramConfig.sr)

    torch.save(model.state_dict(), 'fastspeech_checkpoint')
