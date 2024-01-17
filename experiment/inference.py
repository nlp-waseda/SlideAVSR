import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image

import jiwer
from jiwer import wer
import Levenshtein

import torch
import openai
from module.cloud_vision import detect_text
# from module.easy_ocr import easy_ocr
from module.gpt_filter import gpt_filter

import whisper.whisper as whisper
from whisper.whisper.normalizers import EnglishTextNormalizer
from training.config import Config
from training.trainer import WhisperModelModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action='store_false')
    # data
    parser.add_argument("--video_path", type=str, required=True, help='the path of video files')
    parser.add_argument("--data_path", type=str, required=True, help='the path of dataset (video ids)')
    parser.add_argument("--subtitle_path", type=str, required=True, help='the path of subtitle files')
    parser.add_argument("--data_split", type=str, required=True, help='train, dev, testA, testB')
    # model
    parser.add_argument("--use_api_for_whisper", action='store_true', help='whether to use openai api for inference')
    parser.add_argument("--model_size", type=str, default="large-v3")
    parser.add_argument("--model_path", type=str, help='the path of model checkpoint')
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
    # model parameter
    parser.add_argument("--without_timestamps", action='store_false')
    parser.add_argument("--beam_size", type=int, default=5)
    # module (ocr, gpt)
    parser.add_argument("--ocr", action='store_true',
                        help='use ocr results as prompts, can not be used with image encoder simultaneously')
    parser.add_argument("--ocr_software", default="easy_ocr", choices=["easy_ocr", "cloud_vision"])
    parser.add_argument("--gpt", default="gpt-3.5.turbo", choices=["gpt-3.5.turbo", "gpt-4", "gpt-4-1106-preview"],
                        help='gpt version for summerizing ocr results')
    parser.add_argument("--prompt_num", type=int, default=20, help='max length of gpt summerized prompts')
    parser.add_argument("--use_prepared_prompt", action="store_true",
                        help='use prepared prompt rather than running ocr/gpt at inference time')
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
    # test
    parser.add_argument("--test_mode", action='store_true')
    parser.add_argument("--output_path", type=str)
    return parser.parse_args(sys.argv[1:])


def transcribe(args, cfg, model, audio_path, initial_prompt, image_processor, image_path):
    audio = whisper.load_audio(audio_path)
    if args.use_image:
        if args.vision_encoder == "clip":
            # HACK pad to 18.48-second to fit encoder
            audio = whisper.pad_or_trim(audio, int(18.48 * cfg.sample_rate))
        elif args.vision_encoder == "donut":
            audio = whisper.pad_or_trim(audio, int(29.98 * cfg.sample_rate))
    else:
        audio = whisper.pad_or_trim(audio)
    input_mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    if args.use_image:
        image = Image.open(image_path)
        if args.vision_encoder == "clip":
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        elif args.vision_encoder == "donut":
            image_tensor = image_processor(image, return_tensors="pt").pixel_values[0]
        image_tensor = image_tensor.half().to(model.device)
        image_tensor = image_tensor[None, :]
    else:
        image_tensor = None

    decode_options = whisper.DecodingOptions(
        task="transcribe",
        language="english",
        without_timestamps=args.without_timestamps,
        beam_size=args.beam_size,
        prompt=initial_prompt,
        fp16=False
    )
    result = whisper.decode(model, input_mel, args.use_image, image_tensor, decode_options)
    return result


def main(args, cfg):
    if not args.use_api_for_whisper:
        if args.model_path:
            model = WhisperModelModule(args, cfg, args.model_size)
            state_dict = torch.load(args.model_path)['state_dict']
            model.load_state_dict(state_dict)
            if args.use_image:
                image_processor = model.image_processor
            else:
                image_processor = None
            model = model.model
        else:
            model = whisper.load_model(f"{args.model_size}")
            image_processor = None
    normalizer = EnglishTextNormalizer()
    refs, hypos = [], []

    # if args.ocr and args.ocr_software == "easy_ocr":
    #     import easyocr
    #     ocr_reader = easyocr.Reader(['en'])

    if args.ocr and args.sort_by_wiki_frequency:
        freq_dict = {}
        with open(args.wiki_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                word, freq = line.split(" ")[0], line.split(" ")[1]
                freq_dict[word] = int(freq)

    if args.ocr and args.nearest_neighbor_search:
        whisper_prediction = []
        with open(args.whisper_prediction_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("HYP"):
                    whisper_prediction.append(line.strip()[5:])

    with open(os.path.join(args.data_path, args.data_split + ".txt"), "r") as f:
        video_ids = f.readlines()
        video_ids = [i.strip() for i in video_ids]

    for video_id in tqdm(video_ids):
        with open(os.path.join(args.subtitle_path, f"{video_id}.txt"), "r") as f:
            lines = f.readlines()
            if args.use_prepared_prompt:
                if args.use_ocr_results_as_prompt:
                    with open(os.path.join(args.prompt_path, f"{video_id}_ocr_v2.txt"), "r") as f2:
                        prepared_prompt = f2.readlines()
                else:
                    with open(os.path.join(args.prompt_path, f"{video_id}_gpt_summarized.txt"), "r") as f2:
                        prepared_prompt = f2.readlines()
            else:
                ocr_results_for_save = []
                prompt_for_save = []

            for idx, line in enumerate(lines):
                start = line.split(" ")[2]
                end = line.split(" ")[3]
                midpoint = round((float(start) + float(end)) / 2, 3)
                ref = " ".join(line.split(" ")[5:]).strip()
                ref = normalizer(ref)
                if ref == "":
                    continue
                refs.append(ref)

                if args.ocr:
                    if args.use_prepared_prompt:
                        prompt = prepared_prompt[idx].strip()
                        if prompt != "None" and prompt != "<EMPTY>":
                            word_list = prompt.split(", ")
                            # apply FQ Ranker
                            if args.sort_by_wiki_frequency:
                                array_for_sort = []
                                for word in word_list:
                                    if word in freq_dict:
                                        array_for_sort.append((word, freq_dict[word]))
                                    else:
                                        array_for_sort.append((word, 1))
                                array_for_sort.sort(key=lambda a: a[1])
                                word_list = [t[0] for t in array_for_sort]
                            # apply NNS
                            elif args.nearest_neighbor_search:
                                whisper_hyp = whisper_prediction[idx].strip().split()
                                nn_word_list = []
                                for hyp_word in whisper_hyp:
                                    min_edit_distance = 1e9
                                    min_edit_distance_idx = []
                                    for idx, ocr_result in enumerate(word_list):
                                        edit_distance = Levenshtein.distance(hyp_word, ocr_result)
                                        if edit_distance < min_edit_distance:
                                            min_edit_distance = edit_distance
                                            min_edit_distance_idx = [idx]
                                    for idx in min_edit_distance_idx:
                                        l_idx = max(0, idx - (args.nns_ngram - 1) // 2)
                                        r_idx = min(len(word_list) - 1, idx + (args.nns_ngram - 1) // 2)
                                        if l_idx <= r_idx:
                                            nn_word_list += word_list[l_idx: r_idx + 1]
                                word_list = nn_word_list

                            if len(word_list) > args.max_prompt_num:
                                word_list = word_list[:args.max_prompt_num]
                            initial_prompt = ", ".join(word_list)
                        else:
                            initial_prompt = None
                    else:
                        if args.ocr_software == "easy_ocr":
                            word_list = easy_ocr(
                                ocr_reader,
                                os.path.join(args.video_path, f"en/wav/{video_id}/image2_{midpoint:.3f}.png")
                            )
                        elif args.ocr_software == "cloud_vision":
                            word_list = detect_text(
                                os.path.join(args.video_path, f"en/wav/{video_id}/image2_{midpoint:.3f}.png")
                            )
                        else:
                            raise NotImplementedError
                        if word_list is not None:
                            initial_prompt = gpt_filter(word_list, args.prompt_num, args.gpt)
                            ocr_results_for_save.append(word_list)
                            if initial_prompt is None:
                                prompt_for_save.append("None")
                            else:
                                prompt_for_save.append(initial_prompt)
                        else:
                            initial_prompt = None
                            ocr_results_for_save.append("None")
                            prompt_for_save.append("None")
                else:
                    initial_prompt = None

                if args.use_api_for_whisper:
                    result = openai.Audio.transcribe(
                        file=open(os.path.join(args.video_path, f"en/wav16k/{video_id}/{start}_{end}.wav"), "rb"),
                        model="whisper-1",
                        prompt=initial_prompt,
                    )
                    hypo = normalizer(result["text"])
                else:
                    if args.use_image:
                        image_path = os.path.join(args.video_path, f"en/wav/{video_id}/image2_{midpoint:.3f}.png")
                    else:
                        image_path = None
                    result = transcribe(
                        args,
                        cfg,
                        model,
                        os.path.join(args.video_path, f"en/wav16k/{video_id}/{start}_{end}.wav"),
                        initial_prompt,
                        image_processor,
                        image_path
                    )
                    hypo = normalizer(result.text)

                hypos.append(hypo)

            if args.ocr and not args.use_prepared_prompt:
                with open(os.path.join(args.prompt_path, f"{video_id}_ocr_v2.txt"), "w") as f3:
                    f3.write("\n".join(ocr_results_for_save))
                with open(os.path.join(args.prompt_path, f"{video_id}_gpt_summarized.txt"), "w") as f4:
                    f4.write("\n".join(prompt_for_save))

        if args.test_mode:
            break

    print(f"WER={wer(refs, hypos)}")

    if args.test_mode:
        out = jiwer.process_words(refs, hypos)
        print(jiwer.visualize_alignment(out, skip_correct=False))
    if args.output_path:
        out = jiwer.process_words(refs, hypos)
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "result.txt"), "w") as f5:
            f5.write(jiwer.visualize_alignment(out, skip_correct=False))


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    cfg = Config()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main(args, cfg)
