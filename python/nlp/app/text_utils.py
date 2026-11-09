"""Shared, accent-insensitive text normalization helpers."""

import re
import unicodedata


def normalize(text):
    text = str(text).casefold().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def slug(text):
    return re.sub(r"\s+", "-", normalize(text)).strip("-") or "section"
