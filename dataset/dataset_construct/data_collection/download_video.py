import time
import argparse
import sys
import subprocess
import shutil
import pydub
from pathlib import Path
from util import make_video_url, make_basename, vtt2txt, autovtt2txt
import pandas as pd
from tqdm import tqdm

import asyncio

def parse_args():
  parser = argparse.ArgumentParser(
    description="Downloading videos with subtitle.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("lang",         type=str, help="language code (ja, en, ...)")
  parser.add_argument("sublist",      type=str, help="filename of list of video IDs with subtitles")  
  parser.add_argument("--outdir",     type=str, default="video", help="dirname to save videos")
  parser.add_argument("--keeporg",    action='store_true', default=False, help="keep original audio file.")
  return parser.parse_args(sys.argv[1:])

sem = asyncio.Semaphore(5)

n_ok = 0
n_total = 0

async def download_per_video(lang, videoid, outdir="video", wait_sec=10, keep_org=False):
  async with sem:
    global n_ok, n_total
    fn = {}
    for k in ["wav", "wav16k", "vtt", "txt"]:
      fn[k] = Path(outdir) / lang / k / (make_basename(videoid.strip()) + "." + k[:3])
      fn[k].parent.mkdir(parents=True, exist_ok=True)

    if not fn["wav16k"].exists() or not fn["txt"].exists():
      # download
      url = make_video_url(videoid)
      base = fn["wav"].parent.joinpath(fn["wav"].stem)
      try:
        proc = await asyncio.create_subprocess_shell(f"yt-dlp \
                                                      --sub-lang {lang} \
                                                      --keep-video \
                                                      --match-filter 'duration >= 300 & duration <= 1200' \
                                                      -f 'best[res=720]+bestvideo[vcodec*=avc1]+bestaudio/best' \
                                                      --merge-output-format mp4 \
                                                      --extract-audio \
                                                      --audio-format wav \
                                                      --write-sub \
                                                      --write-description \
                                                      -o {base}.\%\(ext\)s \
                                                      {url}",
                                                      stdout=asyncio.subprocess.PIPE,
                                                      stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
          print(f"Failed to download the video: url = {url}")
          return
      except Exception as e:
        print(f"Failed to download the video: error = {e}")
        return
      try:
        shutil.move(f"{base}.{lang}.vtt", fn["vtt"])
      except Exception as e:
        print(f"Failed to rename subtitle file. The download may have failed: url = {url}, filename = {base}.{lang}.vtt, error = {e}")
        return

      # vtt -> txt (reformatting)
      try:
        txt = vtt2txt(open(fn["vtt"], "r").readlines())
        with open(fn["txt"], "w") as f:
          f.writelines([f"{t[0]:1.3f}\t{t[1]:1.3f}\t\"{t[2]}\"\n" for t in txt])
      except Exception as e:
        print(f"Falied to convert subtitle file to txt file: url = {url}, filename = {fn['vtt']}, error = {e}")
        return

      # wav -> wav16k (resampling to 16kHz, 1ch)
      try:
        wav = pydub.AudioSegment.from_file(fn["wav"], format = "wav")
        wav = pydub.effects.normalize(wav, 5.0).set_frame_rate(16000).set_channels(1)
        wav.export(fn["wav16k"], format="wav", bitrate="16k")
      except Exception as e:
        print(f"Failed to normalize or resample downloaded audio: url = {url}, filename = {fn['wav']}, error = {e}")
        return

      # remove original wav
      if not keep_org:
        fn["wav"].unlink()
        
      n_ok += 1
      if n_ok % 100 == 0:
        print(f"{n_ok}/{n_total} videos are downloaded.")

async def download_video(lang, fn_sub, outdir="video", wait_sec=10, keep_org=False):
  """
  Tips:
    If you want to download automatic subtitles instead of manual subtitles, please change as follows.
      1. replace "sub[sub["sub"]==True]" of for-loop with "sub[sub["auto"]==True]"
      2. replace "--write-sub" option of yt-dlp with "--write-auto-sub"
      3. replace vtt2txt() with autovtt2txt()
      4 (optional). change fn["vtt"] (path to save subtitle) to another. 
  """
  global n_ok, n_total
  
  if fn_sub.endswith(".csv"):
    sub = pd.read_csv(fn_sub)
    tasks = [asyncio.create_task(download_per_video(lang, videoid, outdir, wait_sec, keep_org)) for videoid in tqdm(sub[sub["sub"]==True]["videoid"])]
  else:
    with open(fn_sub) as f:
      videoids = f.readlines()
    tasks = [asyncio.create_task(download_per_video(lang, videoid, outdir, wait_sec, keep_org)) for videoid in tqdm(videoids)]
    
  n_total = len(tasks)
  await asyncio.gather(*tasks)


if __name__ == "__main__":
  args = parse_args()

  asyncio.run(download_video(args.lang, args.sublist, args.outdir, keep_org=args.keeporg))
  
  dirname = Path(args.outdir) / args.lang
  print(f"save {args.lang.upper()} videos to {dirname}.")

