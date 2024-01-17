import os
import sys
import time
import openai
import argparse
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(
	description="Retrieving whether subtitles exists or not.",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("videoidlist",  type=str, help="filename of video ID list")  
	return parser.parse_args(sys.argv[1:])

args = parse_args()
openai.api_key = os.getenv("OPENAI_API_KEY")

path = "video/en/wav/"
with open(args.videoidlist) as f:
	videoids = f.readlines()
f.close()

ok_videoids = []

for id in tqdm(videoids):
	id = id.strip("\n")
	decription_file_path = path + id + "/" + id + ".description"
	with open(decription_file_path) as f:
		description = f.read()
	
	n_total = 0
	n_yes = 0
	while n_total < 3:
		try:
			chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
				{
					"role": "user", 
					"content": f"Here is a description of a YouTube video:\n \
								{description}\n \
								Using the description, check whether the video meets the following criteria.\n \
								- This video is a presentation video of a research paper.\n \
								- The description is written in English.\n \
								Attention, you can only answer 'Yes' or 'No' and you can only answer one time."
				}
			], timeout=5)
			if "Yes" in chat_completion.choices[0]["message"]["content"]:
				n_yes += 1
			n_total += 1
		except Exception as e:
			print(f"Failed, error={e}")
		time.sleep(0.2)
	
	if n_yes >= 1:
		ok_videoids.append(id)

with open("videoid/en/id_for_blip2.txt", "w") as f:
	for id in ok_videoids:
		f.write(f"{id}\n")