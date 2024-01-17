import sys
import argparse

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_align import CTCSegmentation
import soundfile

    
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("wavfile",  type=str)
	parser.add_argument("txtfile",  type=str)
	parser.add_argument("output_dir",  type=str)
	return parser.parse_args(sys.argv[1:])

args = parse_args()

# load a model with character tokens
d = ModelDownloader(cachedir="./modelcache")
wsjmodel = d.download_and_unpack("kamo-naoyuki/wsj")

# load the example file included in the ESPnet repository
speech, rate = soundfile.read(args.wavfile)

# CTC segmentation
aligner = CTCSegmentation( **wsjmodel , fs=rate)
aligner.set_config( gratis_blank=True, kaldi_style_text=False)

with open(args.txtfile, "r") as f:
    text = f.readlines()
new_text = [t.split("\t")[2].strip() for t in text]

segments = aligner(speech, new_text)

with open(args.output_dir + "/segments.txt", "w") as f:
    f.write(str(segments))