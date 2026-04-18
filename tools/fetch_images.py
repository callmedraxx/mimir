#!/usr/bin/env python3
"""
Fetch real photos for the 26 alphabet words directly from Wikimedia
Commons via Special:FilePath (which redirects straight to the CDN —
no REST API, no per-client rate limits).  Convert each to a 128×128
grayscale raw float array in [0,1] for the MIMIR visual sensor.

Output: data/images/<word>.raw  (128*128 = 16384 little-endian floats)
        data/images/<word>.png  (preview)

Raw format/size must match VISION_RAW_SIZE in src/mimir.h.
"""

import os
import struct
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import concurrent.futures
from io import BytesIO
from PIL import Image

IMG_SIZE = 128
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

# Each alphabet word → Wikimedia Commons filename.  Special:FilePath
# resolves these directly to the CDN without the REST API rate limit.
WORD_TO_FILE = {
    'apple':     'Pink_lady_and_cross_section.jpg',
    'ball':      'Football_Pallo_valmiina.jpg',
    'cat':       'Cat_August_2010-4.jpg',
    'dog':       'Huskiesatrest.jpg',
    'egg':       'Huevo_frito.jpg',
    'fish':      'Common_goldfish.JPG',
    'grape':     'Table_grapes_on_white.jpg',
    'hat':       'Fedora_hat.jpg',
    'ice':       'Ice-Cubes.jpg',
    'jam':       'Fruits_jam_variants.jpg',
    'key':       'Sleutel.jpg',
    'lamp':      'Desk_lamp.jpg',
    'moon':      'FullMoon2010.jpg',
    'nest':      'Bird_nest_in_the_garden.jpg',
    'orange':    'Orange-Whole-%26-Split.jpg',
    'pen':       'Carandache_Ecridor.jpg',
    'queen':     'Elizabeth_II_Coronation_Portrait_Herbert_James_Gunn.jpg',
    'rain':      'Rain_on_grass.jpg',
    'sun':       'Sun_in_February_%28black_version%29.jpg',
    'tree':      'Baum_im_Odenwald.jpg',
    'umbrella':  'Umbrella.jpg',
    'van':       'Ford_Transit_Custom_front_20080331.jpg',
    'wagon':     'Horse_drawn_wagon.jpg',
    'xylophone': 'Xylophone.jpg',
    'yarn':      'Yarn_ball.jpg',
    'zebra':     'Plains_Zebra_Equus_quagga.jpg',
}

USER_AGENT = 'MIMIR-fetch/0.2 (research; contact: local)'


def filepath_url(filename: str, width: int = 512) -> str:
    """Build a Special:FilePath URL that redirects to the CDN.
    `?width=N` asks the CDN for a thumbnail (much smaller/faster than
    the full-res original while still high-resolution for our 128×128 crop). """
    return (f'https://commons.wikimedia.org/wiki/Special:FilePath/'
            f'{urllib.parse.quote(filename)}?width={width}')


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as f:
        return f.read()


def get_with_retry(fn, *args, retries=4, base_delay=3.0):
    delay = base_delay
    last = None
    for _ in range(retries):
        try:
            return fn(*args)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 503):
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(delay)
            delay *= 2
    if last:
        raise last


def to_grayscale_square(img: Image.Image, size: int) -> Image.Image:
    img = img.convert('L')
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top  = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def save_raw(img: Image.Image, path: str):
    pixels = [p / 255.0 for p in img.getdata()]
    with open(path, 'wb') as f:
        f.write(struct.pack('<%df' % len(pixels), *pixels))


def fetch_one(word: str, filename: str) -> bool:
    raw_path = os.path.join(OUT_DIR, f'{word}.raw')
    if os.path.exists(raw_path) and os.path.getsize(raw_path) == IMG_SIZE * IMG_SIZE * 4:
        print(f'  {word:12s} already present, skipping', flush=True)
        return True
    try:
        url = filepath_url(filename)
        data = get_with_retry(download, url)
        img = Image.open(BytesIO(data))
        img = to_grayscale_square(img, IMG_SIZE)
        img.save(os.path.join(OUT_DIR, f'{word}.png'))
        save_raw(img, raw_path)
        print(f'  {word:12s} <- {filename}', flush=True)
        return True
    except Exception as e:
        print(f'  {word:12s} FAILED: {e}', flush=True)
        return False


def main():
    print(f'Fetching {len(WORD_TO_FILE)} photos → {IMG_SIZE}x{IMG_SIZE} grayscale raw')
    ok, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, w, f): w for w, f in WORD_TO_FILE.items()}
        for fut in concurrent.futures.as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail += 1
    print(f'\nDone. {ok} ok, {fail} failed.')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
