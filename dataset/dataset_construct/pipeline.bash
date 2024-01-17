# data collection
## generate search words
python3 word/word_list_gen.py

## obtain video ids
python3 data_collection/obtain_video_id.py en word/conf_year_form.txt

## check if manual subtitles are available
python3 data_collection/retrieve_subtitle_exists.py en videoid/en/conf_year_form.txt

## download videos
python3 data_collection/download_video.py en sub/en/conf_year_form.csv

# filtering
## chatgpt filter
find video/en/wav/ -mindepth 1 -type d -exec basename {} \; > videoid/en/id_for_chatgpt.txt
python3 filtering/chatgpt_filter.py videoid/en/id_for_chatgpt.txt

## bilp-2 filter
while read -u 10 p; do
    len_of_video=$(ffprobe video/en/wav/${p}/${p}.mp4 -show_entries format=duration -loglevel quiet | grep 'duration' | sed -e 's/duration\=//')
    len_of_video=${len_of_video%.*}
    for i in {1..5} ; do
        position=$((1 + (len_of_video / 5) * (i - 1)))
        ffmpeg -ss $position -i video/en/wav/${p}/${p}.mp4 -t 1 -r 1 -vcodec png video/en/wav/${p}/image_${i}.png
    done
done 10<videoid/en/id_for_blip2.txt
python3 filtering/blip2_filter.py videoid/en/id_for_blip2.txt

## human filter (you need check the video by yourself or use crowdsourcing)

# cleansing
## num2words
python3 cleansing/num_to_words.py id_v1.0.txt

## ctc segmentation and scoring
while read -u 10 p; do
    wavfile=video/en/wav16k/${p}/${p}.wav
    txtfile=video/en/txt/${p}/${p}_cleaned.txt
    output_dir=video/en/txt/${p}
    python3 cleansing/alignment.py ${wavfile} ${txtfile} ${output_dir}
done 10<id_v1.0.txt

min_confidence_score=-7
while read -u 10 p; do
    awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' video/en/txt/${p}/segments.txt > video/en/txt/${p}/segments_filtered.txt
done 10<id_v1.0.txt

## blip-2 filter
while read -u 10 p; do
    while read line; do
        start=$(echo $line | awk '{print $3}')
        end=$(echo $line | awk '{print $4}')
        midpoint=$(echo "($start + $end) / 2" | bc -l)
        ffmpeg -ss $midpoint -i video/en/wav/${p}/${p}.mp4 -t 1 -r 1 -vcodec png video/en/wav/${p}/image2_${start}.png
    done < video/en/txt/${p}/segments_filtered.txt
done 10<id_v1.0.txt
python3 cleansing/blip2_filter.py id_v1.0.txt