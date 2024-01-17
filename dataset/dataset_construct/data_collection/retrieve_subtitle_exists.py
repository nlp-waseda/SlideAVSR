import time
import requests
import argparse
import re
import sys
import subprocess
from pathlib import Path
from util import make_video_url, get_subtitle_language
import pandas as pd

import asyncio
import traceback

def parse_args():
  parser = argparse.ArgumentParser(
    description="Retrieving whether subtitles exists or not.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("lang",         type=str, help="language code (ja, en, ...)")
  parser.add_argument("videoidlist",  type=str, help="filename of video ID list")  
  parser.add_argument("--outdir",     type=str, default="sub", help="dirname to save results")
  parser.add_argument("--checkpoint", type=str, default=None, help="filename of list checkpoint (for restart retrieving)")
  return parser.parse_args(sys.argv[1:])

sem = asyncio.Semaphore(20)

n_ok = 0
n_total = 0

async def retrieve_subtitle_for_per_video(videoid, lang, wait_sec=0.2):
  async with sem:
    global n_ok, n_total
    videoid = videoid.strip(" ").strip("\n")

    # send query to YouTube
    url = make_video_url(videoid)
    df = None
    try:
      # result = subprocess.check_output(f"yt-dlp --list-subs --sub-lang {lang} --skip-download {url}", shell=True, universal_newlines=True)
      proc = await asyncio.create_subprocess_shell(f"yt-dlp --list-subs --sub-lang {lang} --skip-download {url}",
                                                  stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.PIPE
      )
      stdout, stderr = await proc.communicate()
      auto_lang, manu_lang = get_subtitle_language(stdout.decode())
      df = pd.DataFrame([{"videoid": videoid, "auto": lang in auto_lang, "sub": lang in manu_lang}])
      
      n_ok += 1
      if n_ok % 100 == 0:
        print(f"{n_ok}/{n_total} videos are retrieved.")
    except:
      traceback.print_exc()
      print(f"failed to retrieve subtitle info, video_id={videoid}.")
    
    return df
  
async def retrieve_subtitle_exists(lang, fn_videoid, outdir="sub", wait_sec=0.2, fn_checkpoint=None):
  global n_ok, n_total
  fn_sub = Path(outdir) / lang / f"{Path(fn_videoid).stem}.csv"
  fn_sub.parent.mkdir(parents=True, exist_ok=True)

  # if file exists, load it and restart retrieving.
  if fn_checkpoint is None:
    subtitle_exists = pd.DataFrame({"videoid": [], "auto": [], "sub": []}, dtype=str)
  else:
    subtitle_exists = pd.read_csv(fn_checkpoint)

  # load video ID list
  tasks = [asyncio.create_task(retrieve_subtitle_for_per_video(videoid, lang, wait_sec)) for videoid in open(fn_videoid).readlines()]
  n_total = len(tasks)
  dfs = await asyncio.gather(*tasks)
  for df in dfs:
    if df is not None:
      subtitle_exists = pd.concat([subtitle_exists, df], ignore_index=True)

  # write
  subtitle_exists.to_csv(fn_sub, index=None)

if __name__ == "__main__":
  args = parse_args()

  asyncio.run(retrieve_subtitle_exists(args.lang, args.videoidlist, args.outdir, fn_checkpoint=args.checkpoint))
  
  filename = Path(args.outdir) / args.lang / f"{Path(args.videoidlist).stem}.csv"
  print(f"save {args.lang.upper()} subtitle info to {filename}.")
