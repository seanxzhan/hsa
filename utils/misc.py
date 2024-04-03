import os
import re
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
