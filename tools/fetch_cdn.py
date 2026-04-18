#!/usr/bin/env python3
"""
Final-pass fetcher: build Wikimedia CDN URLs directly from filename
hashes (upload.wikimedia.org/wikipedia/commons/thumb/<hash>/<filename>/512px-<filename>).
This path goes through the image CDN and is not subject to the REST
API rate limit.
"""

import os
import struct
import sys
import time
import hashlib
import urllib.request
import urllib.parse
import urllib.error
from io import BytesIO
from PIL import Image

IMG_SIZE = 128
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
USER_AGENT = 'MIMIR-fetch/0.3 (research; contact: local)'

# Wikimedia Commons filenames we've verified exist for each concept.
# Queen uses the Coronation portrait — shows the crown clearly (per user).
WORD_TO_FILE = {
    'orange':    'Ambersweet_oranges.jpg',
    'queen':     'Elizabeth_II_Coronation_Portrait_Herbert_James_Gunn.jpg',
    'rain':      'Rain_on_grass2.jpg',
    'sun':       'Sun_red_giant.svg',  # svg fallback below to a jpg
    'tree':      'Baum_im_Herbst.jpg',
    'umbrella':  'Opened_umbrella_outside.jpg',
    'van':       'Ford_Transit_front_20080724.jpg',
    'wagon':     'Amishwagon.jpg',
    'xylophone': 'Glockenspiel_or_Xylophone.jpg',
    'yarn':      'Wool_yarn.jpg',
}

# A second, more conservative set of fallbacks — filenames that are
# well-known to exist (they appear on the main Wikipedia article lead).
# Tried if the primary fails.
FALLBACKS = {
    'orange':    ['Orange-Fruit-Pieces.jpg', 'Oranges_-_whole-halved-segment.jpg'],
    'queen':     ['Queen_Elizabeth_II_of_New_Zealand.jpg'],
    'rain':      ['Rain_drops.jpg', 'Regenschirm.jpg'],
    'sun':       ['The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg',
                  'Sunset_2007-1.jpg'],
    'tree':      ['Tree_Fagus_sylvatica.jpg', 'Ash_Tree_-_geograph.org.uk_-_590710.jpg'],
    'umbrella':  ['Umbrella-4587763.jpg', 'Red_Umbrella_Over_White.jpg'],
    'van':       ['1994_Ford_Econoline_E-150.jpg', 'Dodge_Grand_Caravan_--_07-30-2009.jpg'],
    'wagon':     ['Wheeled_vehicle_horse_wagon.jpg', 'Wagon_on_a_Farm.jpg'],
    'xylophone': ['Xylophone_(PSF).jpg', 'Xylophone_3.jpg'],
    'yarn':      ['Yarn.jpg', 'Yarn_colour.jpg'],
}


def cdn_url(filename: str, width: int = 512) -> str:
    """Build upload.wikimedia.org thumbnail URL via MD5 hash convention."""
    # Wikimedia replaces spaces with underscores and uses the MD5 of that.
    name = filename.replace(' ', '_')
    md5 = hashlib.md5(name.encode('utf-8')).hexdigest()
    return (f'https://upload.wikimedia.org/wikipedia/commons/thumb/'
            f'{md5[0]}/{md5[:2]}/{urllib.parse.quote(name)}/'
            f'{width}px-{urllib.parse.quote(name)}')


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as f:
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


def try_filename(word, filename):
    try:
        url = cdn_url(filename)
        data = download(url)
        # .svg thumbs actually return PNG from the CDN, PIL handles it.
        img = Image.open(BytesIO(data))
        img = to_grayscale_square(img, IMG_SIZE)
        img.save(os.path.join(OUT_DIR, f'{word}.png'))
        save_raw(img, os.path.join(OUT_DIR, f'{word}.raw'))
        print(f'  {word:12s} <- {filename}', flush=True)
        return True
    except Exception as e:
        print(f'  {word:12s} tried {filename}: {e}', flush=True)
        return False


def fetch_word(word):
    candidates = [WORD_TO_FILE.get(word)] + FALLBACKS.get(word, [])
    candidates = [c for c in candidates if c]
    for c in candidates:
        if try_filename(word, c):
            return True
        time.sleep(1.0)
    return False


def main():
    missing = []
    for w in WORD_TO_FILE:
        raw = os.path.join(OUT_DIR, f'{w}.raw')
        if not os.path.exists(raw) or os.path.getsize(raw) != IMG_SIZE * IMG_SIZE * 4:
            missing.append(w)
    print(f'{len(missing)} missing: {missing}')
    ok, fail = 0, 0
    for w in missing:
        if fetch_word(w):
            ok += 1
        else:
            fail += 1
        time.sleep(2.0)
    print(f'\nDone. {ok} ok, {fail} failed.')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
