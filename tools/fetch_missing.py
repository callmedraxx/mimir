#!/usr/bin/env python3
"""
Second-pass fetcher: for any of the 26 alphabet words still missing a
.raw file, look up the Wikipedia lead image via the REST summary API
and download it.  Sequential, well-paced, to stay under Wikimedia's
per-client rate limits.
"""

import os
import struct
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import json
from io import BytesIO
from PIL import Image

IMG_SIZE = 128
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
USER_AGENT = 'MIMIR-fetch/0.2 (research; contact: local)'

WORDS = ['apple','ball','cat','dog','egg','fish','grape','hat','ice','jam',
         'key','lamp','moon','nest','orange','pen','queen','rain','sun','tree',
         'umbrella','van','wagon','xylophone','yarn','zebra']

TITLE_FALLBACKS = {
    'ball':     'Ball',
    'egg':      'Egg_as_food',
    'hat':      'Hat',
    'ice':      'Ice_cube',
    'jam':      'Fruit_preserves',
    'key':      'Key_(lock)',
    'lamp':     'Light_fixture',
    'nest':     'Bird_nest',
    'orange':   'Orange_(fruit)',
    'queen':    'Coronation_of_Elizabeth_II',
    'sun':      'Sun',
    'tree':     'Tree',
    'umbrella': 'Umbrella',
    'van':      'Van',
    'wagon':    'Wagon',
    'xylophone':'Xylophone',
    'yarn':     'Yarn',
    'zebra':    'Zebra',
}


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as f:
        return f.read()


def summary_image(title: str) -> str:
    url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}'
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as f:
        data = json.loads(f.read())
    if 'originalimage' in data:
        return data['originalimage']['source']
    if 'thumbnail' in data:
        return data['thumbnail']['source']
    raise RuntimeError(f'no image in summary for {title}')


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


def fetch_word(word, title):
    raw_path = os.path.join(OUT_DIR, f'{word}.raw')
    if os.path.exists(raw_path) and os.path.getsize(raw_path) == IMG_SIZE * IMG_SIZE * 4:
        return True

    for attempt in range(5):
        try:
            url = summary_image(title)
            data = download(url)
            img = Image.open(BytesIO(data))
            img = to_grayscale_square(img, IMG_SIZE)
            img.save(os.path.join(OUT_DIR, f'{word}.png'))
            save_raw(img, raw_path)
            print(f'  {word:12s} <- {title} ({url.rsplit("/",1)[-1][:50]})', flush=True)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                wait = 15 * (2 ** attempt)
                print(f'  {word:12s} rate-limited, waiting {wait}s...', flush=True)
                time.sleep(wait)
                continue
            print(f'  {word:12s} HTTP {e.code}: {e.reason}', flush=True)
            return False
        except Exception as e:
            print(f'  {word:12s} ERROR: {e}', flush=True)
            return False
    return False


def main():
    missing = []
    for w in WORDS:
        raw = os.path.join(OUT_DIR, f'{w}.raw')
        if not os.path.exists(raw) or os.path.getsize(raw) != IMG_SIZE * IMG_SIZE * 4:
            missing.append(w)
    print(f'{len(missing)} missing: {missing}')

    ok, fail = 0, 0
    for w in missing:
        title = TITLE_FALLBACKS.get(w, w.capitalize())
        if fetch_word(w, title):
            ok += 1
        else:
            fail += 1
        time.sleep(5.0)  # pacing between words

    print(f'\nDone. {ok} ok, {fail} failed.')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
