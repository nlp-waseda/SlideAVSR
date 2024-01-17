# script for reproduce SlideAVSR
## download videos
python3 dataset_construct/data_collection/download_video.py en video_ids.txt

## split audio files into utterances
while read -u 10 p; do
    while read -u 11 line; do
        start=$(echo $line | awk '{print $3}')
        end=$(echo $line | awk '{print $4}')
        duration=$(echo $end - $start | bc | awk '{printf "%.2f\n", $0}')
        ffmpeg -n -i video/en/wav16k/${p}/${p}.wav -ss $start -t $duration -c copy video/en/wav16k/${p}/${start}_${end}.wav
    done 11< subtitles/${p}.txt
done 10<video_ids.txt
