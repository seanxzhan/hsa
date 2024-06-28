import os
import re
import colorsys
from PIL import Image


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def anno_id_to_model_id(anno_info_path):
    f = open(anno_info_path, 'r')
    lines = f.readlines()
    f.close()

    out = {}
    for l in lines:
        tokens = l.split(' ')
        out[tokens[0]] = tokens[3]
    
    return out


def save_fig(plt, title, img_path, rotate=False, transparent=False):
    plt.title(title)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=150, 
                transparent=transparent)
    if rotate:
        im = Image.open(img_path)
        im = im.rotate(90)
        im.save(img_path)


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    result = []
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        result.append(l[si:si+(d+1 if i < r else d)])
    return result

def hex2rgb(h):
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def increase_saturation(rgb, percent):
    # convert RGB values to HSV (hue, saturation, value) format
    hsv = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    # increase the saturation by a percentage
    hsv = (hsv[0], hsv[1] + percent / 100.0, hsv[2])
    # convert back to RGB format
    rgb = tuple(max(0, min(int(x * 255), 255)) for x in colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]))
    return rgb

def hex2rgb_sat(h, percent):
    color = hex2rgb(h)
    return increase_saturation(color, percent)

def get_tab_20():
    return [
        hex2rgb('17becf'),
        hex2rgb('dbdb8d'),
        hex2rgb('bcbd22'),
        hex2rgb('c7c7c7'),
        hex2rgb('7f7f7f'),
        hex2rgb('f7b6d2'),
        hex2rgb('e377c2'),
        hex2rgb('c49c94'),
        hex2rgb('8c564b'),
        hex2rgb('c5b0d5'),
        hex2rgb('9467bd'),
        hex2rgb('ff9896'),
        hex2rgb('d62728'),
        hex2rgb('98df8a'),
        hex2rgb('2ca02c'),
        hex2rgb('ffbb78'),
        hex2rgb('ff7f0e'),
        hex2rgb('aec7e8'),
        hex2rgb('1f77b4'),
        hex2rgb('9edae5'),
    ] * 3

def get_tab_20_saturated(percent):
    return [
        hex2rgb_sat('17becf', percent),
        hex2rgb_sat('dbdb8d', percent),
        hex2rgb_sat('bcbd22', percent),
        hex2rgb_sat('c7c7c7', percent),
        hex2rgb_sat('7f7f7f', percent),
        hex2rgb_sat('f7b6d2', percent),
        hex2rgb_sat('e377c2', percent),
        hex2rgb_sat('c49c94', percent),
        hex2rgb_sat('8c564b', percent),
        hex2rgb_sat('c5b0d5', percent),
        hex2rgb_sat('9467bd', percent),
        hex2rgb_sat('ff9896', percent),
        hex2rgb_sat('d62728', percent),
        hex2rgb_sat('98df8a', percent),
        hex2rgb_sat('2ca02c', percent),
        hex2rgb_sat('ffbb78', percent),
        hex2rgb_sat('ff7f0e', percent),
        hex2rgb_sat('aec7e8', percent),
        hex2rgb_sat('1f77b4', percent),
        hex2rgb_sat('9edae5', percent),
    ] * 3
