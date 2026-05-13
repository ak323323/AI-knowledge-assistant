# utils/text_cleaner.py

import re

def clean_text(text: str) -> str:
    """
    Clean noisy PDF text before embedding
    """

    #  Remove timestamps 
    text = re.sub(r"\d{2}/\d{2}/\d{2},?\s*\d{1,2}:\d{2}\s*(AM|PM)", "", text)

    #  Remove file paths
    text = re.sub(r"file:///.*", "", text)

    #  Remove page indicators
    text = re.sub(r"Page \d+ of \d+", "", text)

    #  Remove repeated long dashes / weird chars
    text = re.sub(r"[-–—]{2,}", " ", text)

    #  Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    #  Remove weird encoding artifacts
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    return text.strip()