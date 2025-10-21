from urllib.parse import urlparse

import unicodedata

def clean_unicode(text):
    # Replace problematic characters manually
    text = text.replace('\u2011', '-')  # non-breaking hyphen to normal hyphen
    text = text.replace('\u2013', '-')  # en dash
    text = text.replace('\u2014', '-')  # em dash
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart apostrophes
    text = text.replace('\u201C', '"').replace('\u201D', '"')  # smart quotes
    return text

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except Exception:
        return False


