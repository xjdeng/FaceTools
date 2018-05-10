import Easy_Image as ei
from Easy_Image import gui


import random
import faceSwap as fs
import faceAverage as fa

def faceswap_helper(img, face_img_list):
    img_faces = img.detect_faces()
    if len(img_faces) == 0:
        return img.getimg()
    else:
        bodynum = random.randint(0, len(img_faces) - 1)
    random.shuffle(face_img_list)
    face_img = None
    i = 0
    while (i < len(face_img_list)) & (face_img is None):
        faces = face_img_list[i].detect_faces()
        if len(faces) > 0:
            facenum = random.randint(0, len(faces) - 1)
            face_img = face_img_list[i]
        i += 1
    return fs.swap(face_img, img, facenum, bodynum)
    

def faceswap(img_list, face_img_list, delay = 1):
    gui.slideshow(img_list, delay, faceswap_helper, face_img_list)
    
def faceswap_avg(img_list, avg_face_list, delay = 1):
    faces = []
    for img in avg_face_list:
        faces += img.detect_faces()
    faceswap(img_list, [fa.average(faces)], delay)