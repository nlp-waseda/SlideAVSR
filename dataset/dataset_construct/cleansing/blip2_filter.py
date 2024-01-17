import sys
import time
import argparse
from tqdm import tqdm
import pandas as pd

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

def parse_args():
	parser = argparse.ArgumentParser(
	description="Retrieving whether subtitles exists or not.",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("videoidlist",  type=str, help="filename of video ID list")  
	return parser.parse_args(sys.argv[1:])

args = parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)

video_path = "video/en/wav/"
txt_path = "video/en/txt/"

with open(args.videoidlist) as f:
	videoids = f.readlines()
f.close()

for id in tqdm(videoids):
	id = id.strip("\n")
 
	with open(f"{txt_path}{id}/segments_filtered.txt") as f_in, open(f"{txt_path}{id}/segments_filtered_blip.txt", "w") as f_out:
		lines = f_in.readlines()
		for line in lines:
			start = line.split(" ")[2]
			try:
				raw_image = Image.open(f"{video_path}{id}/image2_{start}.png").convert("RGB")
				image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
				response = model.generate({"image": image, 
										"prompt": "Question: This image is a screenshot of a video, check whether the image meets the following criteria.\n \
													- It is a screen-sharing, not a photo shoot.\n \
													- It is a part of a slide for a research presentation.\n \
													Attention, you can only answer 'Yes' or 'No' and you can only answer one time.\n \
													Answer: "}, num_captions=3)
				for resp in response:
					if "Yes" in resp:
						f_out.write(line)
						break
			except:
				pass