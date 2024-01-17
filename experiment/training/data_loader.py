import numpy as np
from PIL import Image

import torch

import whisper.whisper as whisper
from .utils import load_wave, masking_wave


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, n_mels, n_ctx, ocr) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.ocr = ocr

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        if self.ocr:
            audio_path, text, prompt = self.audio_info_list[id]
        else:
            audio_path, text = self.audio_info_list[id]

        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        if self.ocr and prompt is not None:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            text = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1):]
                + text
            )
            labels = text[1:] + [self.tokenizer.eot]
            # don't calculate loss for prompt part
            sot_index: int = text.index(self.tokenizer.sot)
            for i in range(sot_index + 1):
                labels[i] = -100
        else:
            labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }


class AVSRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_info_list,
        tokenizer,
        sample_rate,
        n_mels,
        vision_encoder,
        image_processor,
        audio_masking
    ) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        self.audio_masking = audio_masking

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        audio_path, text, image_path = self.audio_info_list[id]

        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        if self.audio_masking:
            audio = masking_wave(audio)

        if self.vision_encoder == "clip":
            # HACK pad to 18.48-second to fit encoder
            audio = whisper.pad_or_trim(audio.flatten(), int(18.48 * self.sample_rate))
        elif self.vision_encoder == "donut":
            audio = whisper.pad_or_trim(audio.flatten(), int(29.98 * self.sample_rate))
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        image = Image.open(image_path)
        if self.vision_encoder == "clip":
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        elif self.vision_encoder == "donut":
            image_tensor = self.image_processor(image, return_tensors="pt").pixel_values[0]
        image_tensor = image_tensor.half()

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            "image_tensor": image_tensor
        }


class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


class SightWhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids, image_tensor = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            image_tensor.append(f["image_tensor"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        image_tensor = torch.concat([image_t[None, :] for image_t in image_tensor])

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids
        batch["image_tensor"] = image_tensor

        return batch
