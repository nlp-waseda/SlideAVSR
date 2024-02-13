# SlideAVSR

This is the repository of our paper "SlideAVSR: A Dataset of Paper Explanation Videos for Audio-Visual Speech Recognition".

## Setup
Run the following commands to set up your environment. All of our experiments were running on single A100 (40GB).
```
git clone https://github.com/nlp-waseda/SlideAVSR.git
cd SlideAVSR

python3 -m venv slideavsr-venv
source slideavsr-venv/bin/activate

pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

## Dataset
We built the SlideAVSR dataset Based on [JTubeSpeech](https://github.com/sarulab-speech/jtubespeech), a framework for building audio corpora from YouTube videos. Please refer to `./dataset/dataset_construct/pipeline.bash` for dataset construction details.

Due to the ChatGPT filter and BLIP-2 filter we used in the construction process, it is not easy to reproduce the dataset from zero. We highly recommend using our produced subtitle files for experiments. Processed subtitle files are uploaded in `./dataset/subtitles`.

In order to comply with the YouTube platform's terms of use and copyright policy, we do not release the raw video files. To reproduce our dataset, please run `./dataset/reproduce.sh` to download videos and split audio files into utterances. (If the download process failed, please reduce the number of simultaneous operations in `./dataset/dataset_construct/data_collection/download_video.py`.)

## Experiments
### Whisper with audio inputs only
Run the following commands to train/inference on an audio-only Whisper model. The default model size is set to large-v3, but you can also train other size models.
```
cd experiment

SEED=42
TRAIN_ID="simple-fine-tuning-seed${SEED}"

python3 train.py \
    --video_path PATH_TO/SlideAVSR/dataset/video \
    --data_path PATH_TO/SlideAVSR/dataset/dataset_split \
    --subtitle_path PATH_TO/SlideAVSR/dataset/subtitles \
    --model_size large-v3 \
    --train_id $TRAIN_ID \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --seed $SEED

python3 inference.py \
    --video_path PATH_TO/SlideAVSR/dataset/video \
    --data_path PATH_TO/SlideAVSR/dataset/dataset_split \
    --subtitle_path PATH_TO/SlideAVSR/dataset/subtitles \
    --data_split testA \
    --model_size large-v3 \
    --model_path PATH_TO/SlideAVSR/experiment/content/checkpoints/${TRAIN_ID}/checkpoint-epoch=0000.ckpt \
    --output_path PATH_TO/SlideAVSR/experiment/content/results/${TRAIN_ID}/epoch_00/dev
```

### DocWhisper
Run the following commands to train/inference on DocWhisper. There is some code left over from the prelimenary experiments that is not useful (e.g. use LLaVA as an visual encoder), so please ignore it!

If you want to produce OCR results by yourself, please replace the prompt files in `./experiment/prompt` before start training.
```
SEED=42
MAX_PROMPT=100
TRAIN_ID="prompt-tuning-ocr-seed${SEED}-${MAX_PROMPT}word"

python3 train.py \
    --video_path PATH_TO/SlideAVSR/dataset/video \
    --data_path PATH_TO/SlideAVSR/dataset/dataset_split \
    --subtitle_path PATH_TO/SlideAVSR/dataset/subtitles \
    --model_size large-v3 \
    --train_id $TRAIN_ID \
    --ocr \
    --use_ocr_results_as_prompt \
    --max_prompt_num ${MAX_PROMPT} \
    --prompt_path PATH_TO/SlideAVSR/experiment/prompt \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --seed $SEED

python3 inference.py \
    --video_path PATH_TO/SlideAVSR/dataset/video \
    --data_path PATH_TO/SlideAVSR/dataset/dataset_split \
    --subtitle_path PATH_TO/SlideAVSR/dataset/subtitles \
    --data_split testA \
    --model_size large-v3 \
    --ocr \
    --use_prepared_prompt \
    --use_ocr_results_as_prompt \
    --max_prompt_num ${MAX_PROMPT} \
    --prompt_path PATH_TO/SlideAVSR/experiment/prompt \
    --model_path PATH_TO/SlideAVSR/experiment/content/checkpoints/${TRAIN_ID}/checkpoint-epoch=0000.ckpt \
    --output_path PATH_TO/SlideAVSR/experiment/content/results/${TRAIN_ID}/epoch_00/dev
```

Run the following commands to train/inference on DocWhisper if you want to apply **FQ Ranker**.

```
SEED=42
MAX_PROMPT=100
TRAIN_ID="prompt-tuning-ocr-seed${SEED}-${MAX_PROMPT}word-fqranker"

python3 train.py \
    --video_path PATH_TO/SlideAVSR/dataset/video \
    --data_path PATH_TO/SlideAVSR/dataset/dataset_split \
    --subtitle_path PATH_TO/SlideAVSR/dataset/subtitles \
    --model_size large-v3 \
    --train_id $TRAIN_ID \
    --ocr \
    --use_ocr_results_as_prompt \
    --max_prompt_num ${MAX_PROMPT} \
    --prompt_path PATH_TO/SlideAVSR/experiment/prompt \
    --sort_by_wiki_frequency \
    --wiki_path PATH_TO/SlideAVSR/experiment/enwiki-2023-04-13.txt \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --seed $SEED

python3 inference.py \
    --video_path PATH_TO/SlideAVSR/dataset/video \
    --data_path PATH_TO/SlideAVSR/dataset/dataset_split \
    --subtitle_path PATH_TO/SlideAVSR/dataset/subtitles \
    --data_split testA \
    --model_size large-v3 \
    --ocr \
    --use_prepared_prompt \
    --use_ocr_results_as_prompt \
    --max_prompt_num ${MAX_PROMPT} \
    --prompt_path PATH_TO/SlideAVSR/experiment/prompt \
    --sort_by_wiki_frequency \
    --wiki_path PATH_TO/SlideAVSR/experiment/enwiki-2023-04-13.txt \
    --model_path PATH_TO/SlideAVSR/experiment/content/checkpoints/${TRAIN_ID}/checkpoint-epoch=0000.ckpt \
    --output_path PATH_TO/SlideAVSR/experiment/content/results/${TRAIN_ID}/epoch_00/dev
```

### License
Our code and dataset are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ja) license. Commercial use is prohibited.

### Citation
```
@misc{wang2024slideavsr,
      title={SlideAVSR: A Dataset of Paper Explanation Videos for Audio-Visual Speech Recognition}, 
      author={Hao Wang and Shuhei Kurita and Shuichiro Shimizu and Daisuke Kawahara},
      year={2024},
      eprint={2401.09759},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
