import os
from os.path import join, exists
from os import listdir, makedirs
import shutil
import random

test_count = 14000
for label in ["manipulated_sequences","original_sequences"]:
    if not exists(os.path.join("../data/test" , label)):
        makedirs("../data/test/" + label)


    out_fold = os.path.join("../data/train/" , label)
    images = os.listdir(out_fold)
    randomlist = random.sample(range(0, len(images)-1), test_count)

    for ind in randomlist:
        img_name = images[ind]

        src_path = os.path.join("../data/train",label,img_name)
        dest_path = os.path.join("../data/test" , label)

        shutil.move(src_path , dest_path)






val_count = 10000
for label in ["manipulated_sequences","original_sequences"]:
    if not exists(os.path.join("../data/val" , label)):
        makedirs("../data/val/" + label)


    out_fold = os.path.join("../data/train/" , label)
    images = os.listdir(out_fold)
    randomlist = random.sample(range(0, len(images)-1), val_count)

    for ind in randomlist:
        img_name = images[ind]

        src_path = os.path.join("../data/train",label,img_name)
        dest_path = os.path.join("../data/val" , label)

        shutil.move(src_path , dest_path)