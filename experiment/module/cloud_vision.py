from whisper.whisper.normalizers import EnglishTextNormalizer


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    try:
        response = client.text_detection(
            image=image,
            image_context={'language_hints': ['en']}
        )
        texts = response.text_annotations

        word_list = []
        normalizer = EnglishTextNormalizer()
        for text in texts:
            word_list += normalizer(text.description).split(" ")
        word_list = [s for s in word_list if len(s) > 2]
        word_list = ", ".join(word_list)
        if word_list.count(",") == 0:
            word_list = "<EMPTY>"

        return word_list
    except Exception:
        return "<EMPTY>"
