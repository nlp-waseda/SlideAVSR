import re
import sys
import argparse
from tqdm import tqdm
from num2words import num2words

def parse_args():
	parser = argparse.ArgumentParser(
	description="Retrieving whether subtitles exists or not.",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("videoidlist",  type=str, help="filename of video ID list")  
	return parser.parse_args(sys.argv[1:])

args = parse_args()
with open(args.videoidlist) as f:
	videoids = f.readlines()
f.close()

path = "video/en/txt/"

for id in tqdm(videoids):
	id = id.strip("\n")
	with open(path + id + "/" + id + ".txt", "r") as input, open(path + id + "/" + id + "_cleaned.txt", "w") as output:
		lines = input.readlines()
		for line in lines:
			line = line.replace("&nbsp;", " ").replace(";", "").upper()
			double_quote_index = line.find("\"")
			text = line[double_quote_index + 1:]
			text = " ".join(text.split())
			numbers = re.findall(r"[-+]?(?:\d*\.*\d+)", text)
			for number in numbers:
				text = text.replace(number, num2words(number))
			output.write(line[:double_quote_index] + text.replace("\"", ""))