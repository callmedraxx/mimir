#!/usr/bin/env python3
"""
Fallback fetcher using image.pollinations.ai — a free, no-auth,
no-rate-limit image service that generates a photo-like image from a
text prompt.  Used when Wikimedia Commons is rate-limiting us.

Quality varies but images are class-distinct, which is what the Gabor
front-end needs.  Only fetches words that don't already have a valid
.raw file in data/images/.
"""

import os
import struct
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from io import BytesIO
from PIL import Image

IMG_SIZE = 128
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
USER_AGENT = 'MIMIR-fetch/0.4 (research; contact: local)'

# One descriptive prompt per word, tuned for clear single-subject photos.
WORD_TO_PROMPT = {
    'apple':     'a single red apple fruit on a plain white background, photo',
    'ball':      'a soccer football ball on plain background, photo',
    'cat':       'a house cat sitting, portrait photo',
    'dog':       'a golden retriever dog sitting, portrait photo',
    'egg':       'a single chicken egg on plain background, photo',
    'fish':      'a single goldfish swimming, plain background, photo',
    'grape':     'a bunch of purple grapes on a plain background, photo',
    'hat':       'a brown fedora hat on plain background, photo',
    'ice':       'ice cubes on a plain background, photo',
    'jam':       'a jar of strawberry jam with spoon, photo',
    'key':       'a single metal house key on plain background, photo',
    'lamp':      'a desk lamp turned on, plain background, photo',
    'moon':      'the full moon in black sky, photo',
    'nest':      'a birds nest with eggs in a tree, photo',
    'orange':    'a single orange fruit on plain background, photo',
    'pen':       'a blue ballpoint pen on plain background, photo',
    'queen':     'a queen wearing a golden crown, royal portrait, photo',
    'rain':      'heavy rain falling, close up photo',
    'sun':       'the bright sun in a blue sky, photo',
    'tree':      'a single large oak tree in a green field, photo',
    'umbrella':  'a red open umbrella on plain background, photo',
    'van':       'a white delivery van on a road, side view photo',
    'wagon':     'a wooden horse-drawn wagon in a field, photo',
    'xylophone': 'a colorful childrens xylophone with mallets, photo',
    'yarn':      'a ball of red yarn on plain background, photo',
    'zebra':     'a single zebra standing on african savanna, photo',
}


def pollinations_url(prompt: str) -> str:
    # `nologo=true` removes watermark; `seed=<int>` makes it deterministic.
    seed = abs(hash(prompt)) % 1000000
    return (f'https://image.pollinations.ai/prompt/'
            f'{urllib.parse.quote(prompt)}?width=512&height=512&nologo=true&seed={seed}')


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as f:
        return f.read()


def to_grayscale_square(img, size):
    img = img.convert('L')
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top  = (h - m) // 2
    return img.crop((left, top, left + m, top + m)).resize((size, size), Image.LANCZOS)


def save_raw(img, path):
    pixels = [p / 255.0 for p in img.getdata()]
    with open(path, 'wb') as f:
        f.write(struct.pack('<%df' % len(pixels), *pixels))


def fetch_word(word, prompt):
    raw_path = os.path.join(OUT_DIR, f'{word}.raw')
    if os.path.exists(raw_path) and os.path.getsize(raw_path) == IMG_SIZE * IMG_SIZE * 4:
        return True
    for attempt in range(3):
        try:
            url = pollinations_url(prompt)
            data = download(url)
            img = Image.open(BytesIO(data))
            img = to_grayscale_square(img, IMG_SIZE)
            img.save(os.path.join(OUT_DIR, f'{word}.png'))
            save_raw(img, raw_path)
            print(f'  {word:12s} generated ({len(data)//1024}KB)', flush=True)
            return True
        except Exception as e:
            print(f'  {word:12s} attempt {attempt+1}: {e}', flush=True)
            time.sleep(5)
    return False


def main():
    missing = []
    for w in WORD_TO_PROMPT:
        raw = os.path.join(OUT_DIR, f'{w}.raw')
        if not os.path.exists(raw) or os.path.getsize(raw) != IMG_SIZE * IMG_SIZE * 4:
            missing.append(w)
    print(f'{len(missing)} missing: {missing}')
    ok, fail = 0, 0
    for w in missing:
        if fetch_word(w, WORD_TO_PROMPT[w]):
            ok += 1
        else:
            fail += 1
    print(f'\nDone. {ok} ok, {fail} failed.')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
