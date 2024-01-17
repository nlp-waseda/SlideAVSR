import torch
from torch import nn
from pytorch_lightning import LightningModule
from transformers import AdamW, get_linear_schedule_with_warmup

import jiwer

import whisper.whisper as whisper
from whisper.whisper.normalizers import EnglishTextNormalizer
from .data_loader import ASRDataset, AVSRDataset, WhisperDataCollatorWhithPadding, SightWhisperDataCollatorWhithPadding
from .model import load_model as sightwhisper_load_model


class WhisperModelModule(LightningModule):
    def __init__(
        self,
        args,
        cfg,
        model_name="large-v3",
        lang="en",
        train_dataset=[],
        eval_dataset=[]
    ) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)

        self.use_image = args.use_image
        if self.use_image:
            self.model, self.image_processor = sightwhisper_load_model(args, model_name)
            self.vision_encoder = args.vision_encoder
        else:
            self.model = whisper.load_model(model_name)
        self.audio_masking = args.audio_masking

        self.ocr = args.ocr

        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)
        self.normalizer = EnglishTextNormalizer()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            if self.use_image:
                audio_features = self.model.encoder(input_ids, batch["image_tensor"])
            else:
                audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        if self.use_image:
            audio_features = self.model.encoder(input_ids, batch["image_tensor"])
        else:
            audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.normalizer(self.tokenizer.decode(o)))
            l_list.append(self.normalizer(self.tokenizer.decode(l)))

        wer = jiwer.wer(l_list, o_list)
        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.learning_rate,
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        if self.use_image:
            dataset = AVSRDataset(
                self.__train_dataset,
                self.tokenizer,
                self.cfg.sample_rate,
                self.model.dims.n_mels,
                self.vision_encoder,
                self.image_processor,
                self.audio_masking
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                collate_fn=SightWhisperDataCollatorWhithPadding()
            )
        else:
            dataset = ASRDataset(
                self.__train_dataset,
                self.tokenizer,
                self.cfg.sample_rate,
                self.model.dims.n_mels,
                self.model.dims.n_text_ctx,
                self.ocr
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                collate_fn=WhisperDataCollatorWhithPadding()
            )

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        if self.use_image:
            dataset = AVSRDataset(
                self.__eval_dataset,
                self.tokenizer,
                self.cfg.sample_rate,
                self.model.dims.n_mels,
                self.vision_encoder,
                self.image_processor,
                False
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_worker,
                collate_fn=SightWhisperDataCollatorWhithPadding()
            )
        else:
            dataset = ASRDataset(
                self.__eval_dataset,
                self.tokenizer,
                self.cfg.sample_rate,
                self.model.dims.n_mels,
                self.model.dims.n_text_ctx,
                self.ocr
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_worker,
                collate_fn=WhisperDataCollatorWhithPadding()
            )
