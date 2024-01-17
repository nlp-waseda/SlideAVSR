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

path = "video/en/wav/"
with open(args.videoidlist) as f:
	videoids = f.readlines()
f.close()

ok_videoids = []

for id in tqdm(videoids):
	id = id.strip("\n")
	n_yes = 0
	file_not_found = False
	for i in range(1, 5 + 1):
		try:
			raw_image = Image.open(f"video/en/wav/{id}/image_{i}.png").convert("RGB")
		except:
			file_not_found = True
			break
		image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
		response = model.generate({"image": image, 
                             	  "prompt": "Question: This image is a screenshot of a video, check whether the image meets the following criteria.\n \
											- It is a screen-sharing, not a photo shoot.\n \
											- It is a part of a slide for a research presentation.\n \
											Attention, you can only answer 'Yes' or 'No' and you can only answer one time.\n \
											Answer: "})
		if "Yes" in response[0]:
			n_yes += 1
	if file_not_found:
		continue
	if n_yes >= 1:
		ok_videoids.append(id)

def make_video_url(videoid: str) -> str:
    return f"https://www.youtube.com/watch?v={videoid}"

df = pd.DataFrame({"videoid": ok_videoids, "url": [make_video_url(id) for id in ok_videoids]})
df.to_csv("videoid/en/id_for_human.csv")