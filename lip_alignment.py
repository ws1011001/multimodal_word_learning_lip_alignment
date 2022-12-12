#!/usr/bin/env python
# coding: utf-8


## Load up packages
import os
import dlib  # use the pre-trained predictor in dlib for detecting facial landmarks, see # http://dlib.net/face_landmark_detection.py.html

import numpy as np
import pandas as pd

from tqdm import tqdm
from matplotlib import pyplot as plt
from moviepy.editor import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


## Setup working path
dir_project = '/Projects/multimodal_word_learning'
dir_stimuli = os.path.join(dir_project, 'experiment_design/stimuli')
f_predictor = os.path.join(dir_stimuli, 'shape_predictor_68_face_landmarks.dat')


## Utils
def detect_movements(img_dir, img_name, predictor_path, n_img):
    '''
    Function to detect horizontal and vertical lip movements by using dlib's 68-face-landmarks predictor.
    '''
    # Load up predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # Detect facial landmarks for each frame
    mv_v = np.zeros(n_img)  # vertical lip movements
    mv_h = np.zeros(n_img)  # horizontal lip movements
    for i in range(n_img):
        f_img = os.path.join(img_dir, img_name, f'{img_name}_out-{i+1:04d}.png')
        img = dlib.load_rgb_image(f_img)
        box = detector(img, 1)
        shape = predictor(img, box[0])
        # Calculate vertical and horizontal lip movements
        mv_v[i] = np.abs(shape.part(57).y - shape.part(51).y)
        mv_h[i] = np.abs(shape.part(64).x - shape.part(60).x)
        
    return mv_v, mv_h

def detect_shift(shape, ref_shape=None):
    shape = shape_vector(shape)
    center = shape[48:, :].mean(axis=0)  # center of mass
    ref_shape = shape_vector(ref_shape)
    print(f'The shape center is {center}.')
    if ref_shape is not None:
        ref_center = ref_shape[48:, :].mean(axis=0)
        x_shift = np.round(ref_center[0] - center[0])  # at least 1 pixel
        y_shift = np.round(ref_center[1] - center[1])
        print(f'The reference is {ref_center}. X shift {x_shift}, Y shift {y_shift}.')
    else:
        x_shift, y_shift = 0, 0
        
    return center, x_shift, y_shift

def shape_vector(shape):
    V = np.zeros((68, 2))
    for i in range(68):
        V[i, 0] = shape.part(i).x  # x coordinate
        V[i, 1] = shape.part(i).y  # y coordinate
    
    return V


## Read up video information
df_vids = pd.read_csv(os.path.join(dir_stimuli, 'stimuli_videos.csv'))
dur_video = 1760  # video duration in ms
dur_frame = 5     # frame resolution


## Clip videos according to the onset and offset of lip-movements (i.e., articulation)
df_vids['movement_vertical'] = [[] for _ in range(len(df_vids))]
df_vids['movement_horizontal'] = [[] for _ in range(len(df_vids))]
for i in tqdm(range(len(df_vids))):
    i_stim = df_vids.loc[i, 'stimulus']  # video name
    print(f"Clip the video {i_stim}.")
    # Detect movements
    v, h = detect_movements(os.path.join(dir_stimuli, 'videos'), i_stim, f_predictor)  
    df_vids.at[i, 'movement_vertical'] = list(v)
    df_vids.at[i, 'movement_horizontal'] = list(h)
    # Plot movements
    T = np.linspace(0, dur_video, np.ceil(dur_video / dur_frame))
    aud_onset = np.floor(df_vids.loc[i, 'audio_onset'] / dur_frame)
    aud_offset = np.ceil(df_vids.loc[i, 'audio_offset'] / dur_frame)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(T, v, 'r', label='Vertical')    # vertical movements
    ax.plot(T, h, 'b', label='Horizontal')  # horizontal movements
    ax.plot(aud_onset * dur_frame, v[int(aud_onset)], 'Xg', ms=8)    # audio onset mark
    ax.plot(aud_offset * dur_frame, v[int(aud_offset)], 'Xg', ms=8)  # audio offset mark
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pixels')
    ax.legend()
    fig.patch.set_facecolor('white')
    plt.savefig(os.path.join(dir_stimuli, 'videos', f'{i_stim}_movements.png'), dpi=300, transparent=False)

# Output video information: have to manually identify the onset and offset of lip-movements/articulation
df_vids.to_csv(os.path.join(dir_stimuli, 'stimuli_videos.csv'), index=False)

# Cut out videos
df_vids = pd.read_csv(os.path.join(dir_stimuli, 'stimuli_videos.csv'))  # with the manully identified onsets and offsets
if 'video_init_dur' not in df_vids.columns:
    df_vids['video_init_dur'] = np.zeros(len(df_vids))
    df_vids['video_cut_dur'] = np.zeros(len(df_vids))
for i in tqdm(range(len(df_vids))):
    i_stim = df_vids.loc[i, 'stimulus']
    f_video = os.path.join(dir_stimuli, 'videos', f'{i_stim}_slient.avi')
    vid = VideoFileClip(f_video)
    vid_onset = (df_vids.loc[i, 'video_onset']-1) * dur_frame/1000
    vid_offset = (df_vids.loc[i, 'video_offset']-1) * dur_frame/1000 + 0.15  # extend the current offset by 150ms
    if vid_offset > dur_video: vid_offset = dur_video                        # maximal duration
    vid_cut = vid.subclip(vid_onset, vid_offset)
    vid_cut.write_videofile(os.path.join(dir_stimuli, 'videos', f'{i_stim}_silent_cut.avi'), codec='libx264')
    df_vids.loc[i, 'video_init_dur'] = vid.duration
    df_vids.loc[i, 'video_cut_dur'] = vid_cut.duration
    
# Output video information
df_vids.to_csv(os.path.join(dir_stimuli, 'stimuli_videos.csv'), index=False)


## Align videos
# Load up predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f_predictor)
# Crop and align videos according to the mouth position 
if 'video_x_center' not in df_vids.columns:
    df_vids['video_x_center'] = np.zeros(len(df_vids))
    df_vids['video_y_center'] = np.zeros(len(df_vids))
for i in tqdm(range(len(df_vids))):
    # Extract facial landmarks for this video
    i_stim = df_vids.loc[i, 'stimulus']
    print(f"Crop the video {i_stim}.")
    idx = df_vids.loc[i, 'video_onset']
    frm = os.path.join(dir_stimuli, 'videos', i_stim, f'{i_stim}_out-{idx:04d}.png')  # reference frame
    img = dlib.load_rgb_image(frm)
    box = detector(img, 1)
    shape = predictor(img, box[0])
    # Calculate the center of mass (lip)
    center = detect_shift(shape)
    df_vids.loc[i, 'video_x_center'] = center[0]
    df_vids.loc[i, 'video_y_center'] = center[1]
    # Crop this video
    f_video = os.path.join(dir_stimuli, 'videos', f'{i_stim}_silent_cut.avi')
    vid = VideoFileClip(f_video)
    vid_lip = vid.crop(x1=center[0] - 250, y1=center[1] - 100, 
                       x2=center[0] + 250, y2=center[1] + 110)  # align to the center of the lip
    vid_lip.write_videofile(os.path.join(dir_stimuli, 'videos', f'{i_stim}_lip.avi'), codec='libx264')


# Output video information
df_vids.to_csv(os.path.join(dir_stimuli, 'stimuli_videos.csv'), index=False)

