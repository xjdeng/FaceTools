import Easy_Image as ei
from Easy_Image import gui


import random
import faceSwap as fs

def faceswap_helper(img, face_list):
    img_faces = img.detect_faces()
    if len(img_faces) == 0:
        return img.getimg()
    else:
        bodynum = random.randint(0, len(img_faces) - 1)
    random.shuffle(face_list)
    face_img = None
    i = 0
    while (i < len(face_list)) & (face_img is None):
        faces = face_img[i].detect_faces()
        if len(faces) > 0:
            facenum = random.randint(0, len(faces) - 1)
            face_img = face_img[i]
        i += 1
    return fs.swap(face_img, img, facenum, bodynum)
    

def faceswap(img_list, face_img_list, delay = 1):
    