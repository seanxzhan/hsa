import os
import re


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
