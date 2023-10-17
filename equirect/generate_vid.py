"""
generate video from images in test/ folder
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # get all imgs in test/ in the format rgb_{i}.png
    imgs = []
    path = 'all_poses'
    files = os.listdir(path)
    for file in files:
        if file.startswith('rgb_'):
            imgs.append(file)
    
    imgs.sort()

    # get dims from first img
    img = cv2.imread(os.path.join(path, imgs[0]))
    H, W, _ = img.shape

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{path}.avi', fourcc, 10.0, (W, H))
    # now write all imgs to video
    for img in imgs:
        out.write(cv2.imread(os.path.join(path, img)))
    out.release()
    cv2.destroyAllWindows()
    print('Done')