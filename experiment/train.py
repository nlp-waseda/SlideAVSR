import os
import sys
import argparse
import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from training.config import Config
from training.trainer import WhisperModelModule
from training.utils import get_audio_file_list, get_audio_image_file_list


SAMPLE_RATE = 16000
TEXT_MAX_LENGTH = 240
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action='store_true')
    # data
    parser.add_argument("--video_path", type=str, required=True, help='the path of video files')
    parser.add_argument("--data_path", type=str, required=True, help='the path of dataset (video ids)')
    parser.add_argument("--subtitle_path", type=str, required=True, help='the path of subtitle files')
    # model
    parser.add_argument("--model_size", type=str, default="large-v3")
    parser.add_argument("--use_image", action='store_true', help='whether to use image encoder')
    parser.add_argument("--vision_encoder", default="clip", choices=["clip", "donut"])
    parser.add_argument("--no_llava_mlp", action='store_true',
                        help="do not use llava's mlp layer (only a linear layer for dimension projection)")
    parser.add_argument("--audio_masking", action='store_true', help='mask a part of audio for robust training')
    # lora
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_clip", action='store_true')
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=16)
    # trainer
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    # prompt tuning
    parser.add_argument("--ocr", action='store_true',
                        help='use ocr results as prompts, can not be used with image encoder simultaneously')
    parser.add_argument("--use_ocr_results_as_prompt", action="store_true",
                        help='do not use gpt for summerizing, just use raw ocr results for prompts')
    parser.add_argument("--max_prompt_num", type=int, default=50, help='max prompt length given to whisper')
    parser.add_argument("--sort_by_wiki_frequency", action="store_true", help='sort ocr results by word frequency')
    parser.add_argument("--wiki_path", type=str)
    parser.add_argument("--nearest_neighbor_search", action="store_true",
                        help='use audio only whisper predictions to find the nearest neighbor words in ocr results')
    parser.add_argument("--nns_ngram", type=int, default=5,
                        help='when found a nns word, append how many gram into prompts')
    parser.add_argument("--whisper_prediction_file", type=str,
                        help='the predicition file of audio only whisper')
    parser.add_argument("--prompt_path", type=str, help='the directory of the processed prompt files')
    # others
    parser.add_argument("--train_id", type=str, default=str(datetime.datetime.now()))
    return parser.parse_args(sys.argv[1:])


def main(args):
    if args.use_image:
        if args.vision_encoder == "clip":
            audio_max_length = 295680  # HACK 18.48-second to fit encoder
        elif args.vision_encoder == "donut":
            audio_max_length = 479680
        train_audio_transcript_pair_list = get_audio_image_file_list(
            args,
            "train",
            TEXT_MAX_LENGTH,
            audio_max_length,
            SAMPLE_RATE,
        )
        eval_audio_transcript_pair_list = get_audio_image_file_list(
            args,
            "dev",
            TEXT_MAX_LENGTH,
            audio_max_length,
            SAMPLE_RATE
        )
    else:
        audio_max_length = 480000
        train_audio_transcript_pair_list = get_audio_file_list(
            args,
            "train",
            TEXT_MAX_LENGTH,
            audio_max_length,
            SAMPLE_RATE
        )
        eval_audio_transcript_pair_list = get_audio_file_list(
            args,
            "dev",
            TEXT_MAX_LENGTH,
            audio_max_length,
            SAMPLE_RATE
        )

    print("TRAIN AUDIO DATASET NUM: ", len(train_audio_transcript_pair_list))
    print("EVAL AUDIO DATASET NUM: ", len(eval_audio_transcript_pair_list))

    log_output_dir = "./content/logs"
    check_output_dir = "./content/checkpoints/"
    os.makedirs(os.path.dirname(log_output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(check_output_dir), exist_ok=True)
    train_name = "whisper"
    model_name = args.model_size
    lang = "en"

    cfg = Config()
    cfg.learning_rate = args.learning_rate
    cfg.warmup_steps = args.warmup_steps
    cfg.batch_size = args.batch_size
    cfg.num_train_epochs = args.num_train_epochs
    cfg.gradient_accumulation_steps = args.gradient_accumulation_steps

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=args.train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{args.train_id}",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=1
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(
        args,
        cfg,
        model_name,
        lang,
        train_audio_transcript_pair_list,
        eval_audio_transcript_pair_list
    )

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        devices="auto",
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )

    trainer.fit(model)


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    seed_everything(args.seed, workers=True)
    main(args)
