import os
from os.path import join, exists
from os import listdir, makedirs
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
from skimage.io import imsave
import imageio.core.util

import warnings
warnings.filterwarnings("ignore")

def ignore_warnings(*args, **kwargs):
    pass
imageio.core.util._precision_warn = ignore_warnings
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
)



def process(frame,filename):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    face = mtcnn(frame)
    try:
        imsave(
            filename,
            face.permute(1, 2, 0).int().numpy(),
        )
    except AttributeError:
        print("Image skipping")



def save_frame(vid_path, video, counter, label):
    if label == "manipulated_sequences":
        cap = cv2.VideoCapture(vid_path)
        ct = 0
        while cap.isOpened():
            frameId = cap.get(1)
            ret, frame = cap.read()
            if ret == False:
                break
            if frameId % 5 == 0:
                ct = ct + 1
                filename = os.path.join("../data/df_frames/", label, video[:-4]) + str(int(frameId)) + str(counter) + ".jpg"
                process(frame,filename)
            
            if ct == 20:           # only ct frames from 1 video
                break
        cap.release()

    else:
        cap = cv2.VideoCapture(vid_path)
        ct = 0
        while cap.isOpened():
            frameId = cap.get(1) 
            ret, frame = cap.read()
            if ret == False:
                break
            ct = ct + 1
            filename = os.path.join("../data/df_frames/", label, video[:-4]) + str(int(frameId)) + str(counter) + ".jpg"
            process(frame,filename)
            
            if ct == 100:           # only ct frames from 1 video
                break
        cap.release()


counter = 0
for label in ["manipulated_sequences","original_sequences"]:
    out_fold = os.path.join("../data/downloaded/" , label)
    print(os.listdir(out_fold))

    for inner_fold in os.listdir(out_fold):
        videos_path = os.path.join(out_fold , inner_fold , "c40" , "videos")
        if not exists(os.path.join("../data/df_frames" , label)):
            makedirs("../data/df_frames/" + label)
            
        for video in os.listdir(videos_path):
            vid_path = os.path.join(videos_path , video)

            if video[-3:] == "mp4":
                save_frame(vid_path, video, counter, label)

            if counter % 100 == 0:
                print("Number of videos done:", counter)
            counter += 1