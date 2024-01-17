import os
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.transforms as at

from whisper.whisper.normalizers import EnglishTextNormalizer
import Levenshtein


def load_wave(wave_path, sample_rate=16000):
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


# apply masking to waveform, p is the maximum proportion of masking
def masking_wave(wave, p=0.2):
    length = wave.shape[1]
    mask_len = torch.rand(1)[0] * length * p
    start = torch.rand(1)[0] * (length - mask_len)
    start = start.long()
    end = start.long() + mask_len.long()
    mask = torch.arange(0, length)
    wave = wave.masked_fill((mask >= start) & (mask < end), 0)
    return wave


# get dataset info
def get_audio_file_list(
    args,
    data_split="train",
    text_max_length=120,
    audio_max_sample_length=480000,
    sample_rate=16000
):
    audio_transcript_pair_list = []
    normalizer = EnglishTextNormalizer()

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

    with open(os.path.join(args.data_path, data_split + ".txt"), "r") as f:
        video_ids = f.readlines()
        video_ids = [i.strip() for i in video_ids]

    for video_id in tqdm(video_ids):
        with open(os.path.join(args.subtitle_path, f"{video_id}.txt"), "r") as f:
            lines = f.readlines()
            if args.ocr:
                if args.use_ocr_results_as_prompt:
                    with open(os.path.join(args.prompt_path, f"{video_id}_ocr_v2.txt"), "r") as f2:
                        prepared_prompt = f2.readlines()
                else:
                    with open(os.path.join(args.prompt_path, f"{video_id}_gpt_summarized.txt"), "r") as f2:
                        prepared_prompt = f2.readlines()
            for idx, line in enumerate(lines):
                start = line.split(" ")[2]
                end = line.split(" ")[3]
                ref = " ".join(line.split(" ")[5:]).strip()
                ref = normalizer(ref)
                audio_path = os.path.join(args.video_path, f"en/wav16k/{video_id}/{start}_{end}.wav")
                audio = load_wave(audio_path, sample_rate=sample_rate)[0]
                if ref == "" or len(ref) > text_max_length or len(audio) > audio_max_sample_length:
                    continue

                if args.ocr:
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
                    audio_transcript_pair_list.append((audio_path, ref, initial_prompt))
                else:
                    audio_transcript_pair_list.append((audio_path, ref))

    return audio_transcript_pair_list


def get_audio_image_file_list(
    args,
    data_split="train",
    text_max_length=120,
    audio_max_sample_length=480000,
    sample_rate=16000
):
    audio_transcript_pair_list = []
    normalizer = EnglishTextNormalizer()

    with open(os.path.join(args.data_path, data_split + ".txt"), "r") as f:
        video_ids = f.readlines()
        video_ids = [i.strip() for i in video_ids]

    for video_id in tqdm(video_ids):
        with open(os.path.join(args.video_path, f"en/txt/{video_id}/segments_filtered_blip_merged.txt"), "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                start = line.split(" ")[2]
                end = line.split(" ")[3]
                midpoint = round((float(start) + float(end)) / 2, 3)
                ref = " ".join(line.split(" ")[5:]).strip()
                ref = normalizer(ref)
                audio_path = os.path.join(args.video_path, f"en/wav16k/{video_id}/{start}_{end}.wav")
                audio = load_wave(audio_path, sample_rate=sample_rate)[0]
                if ref == "" or len(ref) > text_max_length or len(audio) > audio_max_sample_length:
                    continue
                image_path = os.path.join(args.video_path, f"en/wav/{video_id}/image2_{midpoint:.3f}.png")
                audio_transcript_pair_list.append((audio_path, ref, image_path))

    return audio_transcript_pair_list
