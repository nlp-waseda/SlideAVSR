from whisper.whisper.normalizers import EnglishTextNormalizer


def easy_ocr(ocr_reader, image_path):
    ocr_results = ocr_reader.readtext(
        image_path,
        detail=0,
        text_threshold=0.85
    )
    word_list = []
    normalizer = EnglishTextNormalizer()
    for words in ocr_results:
        word_list += normalizer(words).split(" ")
    word_list = list(set(word_list))
    word_list = [s for s in word_list if len(s) >= 2]
    word_list = ", ".join(word_list)
    if word_list.count(",") == 0:
        word_list = "<EMPTY>"
    return word_list
