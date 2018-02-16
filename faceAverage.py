#!/usr/bin/env python

# Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
# All rights reserved. No warranty, explicit or implicit, provided.


import os
import cv2
import numpy as np
from path import Path as path
import math
import random
import dirtools
import Easy_Facial_Recognition as efr

# Read points from text files in directory
def readPoints(path) :
    # Create an array of array of points.
    pointsArray = [];

    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        
        if filePath.endswith(".txt"):
            
            #Create an array of points.
            points = [];            
            
            # Read points from filePath
            with open(os.path.join(path, filePath)) as file :
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y)))
            
            # Store array of points
            pointsArray.append(points)
            
    return pointsArray;

# Read all jpg images in folder.
def readImages(path) :
    
    #Create array of array of images.
    imagesArray = [];
    
    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith(".jpg"):
            # Read image found.
            img = cv2.imread(os.path.join(path,filePath));

            # Convert to floating point
            img = np.float32(img)/255.0;

            # Add to array of images
            imagesArray.append(img);
            
    return imagesArray;
                
# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform;


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect);
   
    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]));

   
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList();

    # Find the indices of triangles in the points array

    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        

    
    return delaunayTri


def constrainPoint(p, w, h) :
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p;

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def random_avg_dir(face_dir, dims = "", n = 20):
    """
Get the average of n faces found under in face_dir
    """
    results = []
    p = path(face_dir)
    files = p.files()
    order = random.sample(files, len(files))
    i = 0
    while (i < len(files)) & (len(results) < n):
        try:
            img = efr.EasyImageFile(order[i])
            faces = img.detect_faces()
            results += faces
        except efr.NotAnImage:
            pass
        i += 1
    return average(results[0:min(n, len(results))], dims)

def random_avg_subdir(myinput, dims = "", n = 20):
    """
Get the average of n faces chosen from different subdirectories of myinput. 
You can also pass a list of directories into myinput instead of a string
representing the target directory
    """
    results = []
    if isinstance(myinput, list) == False:
        subdirs = dirtools.all_subdirs(myinput)
    else:
        subdirs = myinput
    order = random.sample(subdirs, len(subdirs))
    i = 0
    while (i < len(order)) & (len(results) < n):
        d = path(order[i])
        files = d.files()
        j = 0
        goahead = False
        order2 = random.sample(files, len(files))
        while (j < len(files)) & (len(results) < n) & (goahead == False):
            try:
                img = efr.EasyImageFile(order2[j])
                faces = img.detect_faces()
                results += faces
            except efr.NotAnImage:
                pass
            j += 1
        i += 1
    return average(results[0:min(n, len(results))], dims)
            
    


def weighted_avg_dir(face_dir, dims = "", weights = None, resolution = 10):
    face_list = efr.faces_in_dir(face_dir)
    return weighted_avg(face_list, dims, weights, resolution)

def weighted_avg(face_list, dims = "", weights = None, resolution = 10):  
    """
Like average() but with a bias vector included in the weights variable as a 
list of integers which generates a random average face.  If no weights are 
specified, then a random weight vector will be generated using integers 
between 0 and resolution
    """
    if weights is None:
        weights = [random.randint(0,resolution) for f in face_list]
    if len(weights) < len(face_list):
        e = Exception("Not enough weights specified for your list of faces.")
        raise(e)
    new_face_list = []
    for i in range(0, len(weights)):
        new_face_list += int(round(weights[i]))*[face_list[i]]
    return average(new_face_list, dims)
    
    
    
def average_dir(face_dir, dims = ""):
    """
Gets the average of all faces in directory face_dir
    """
    face_list = efr.faces_in_dir(face_dir)
    return average(face_list, dims)

def average(face_list, dims = ""):
    """
Gets the average of all faces in a face_list consisting of EasyFace objects.
    """
    if len(face_list) == 0:
        return None
    if isinstance(dims, tuple):
        h = dims[0]
        w = dims[1]
    else:
        face_rects = [y.face for y in face_list]
        widths = [x.right() - x.left() for x in face_rects]
        heights = [x.bottom() - x.top() for x in face_rects]
        if dims.lower() == "max":
            w = max(widths)
            h = max(heights)
        elif dims.lower() == "min":
            w = min(widths)
            h = min(heights)
        else: #Use the average widths and heights of faces in face_list
            w = int(round(np.mean(widths)))
            h = int(round(np.mean(heights)))
    allPoints = [f.getpoints() for f in face_list]
    #Get floating point image data:
    images = [np.float32(f.parent_image.getimg())/255.0 for f in face_list]
    # Eye corners
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]
    
    imagesNorm = []
    pointsNorm = []
    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]);
    # Initialize location of average points to 0s
    pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32());
    
    numImages = len(images)
    
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    
    for i in range(0, numImages):

        points1 = allPoints[i];

        # Corners of the eye in input image
        eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] ;
        
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst);
        
        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w,h));

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68,1,2));        
        
        points = cv2.transform(points2, tform);
        
        points = np.float32(np.reshape(points, (68, 2)));
        
        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)
        
        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages;
        
        pointsNorm.append(points);
        imagesNorm.append(img);
    

    
    # Delaunay triangulation
    rect = (0, 0, w, h);
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg));

    # Output image
    output = np.zeros((h,w,3), np.float32());

    # Warp input images to average image landmarks
    for i in range(0, len(imagesNorm)) :
        img = np.zeros((h,w,3), np.float32());
        # Transform triangles one by one
        for j in range(0, len(dt)) :
            tin = []; 
            tout = [];
            
            for k in range(0, 3) :                
                pIn = pointsNorm[i][dt[j][k]];
                pIn = constrainPoint(pIn, w, h);
                
                pOut = pointsAvg[dt[j][k]];
                pOut = constrainPoint(pOut, w, h);
                
                tin.append(pIn);
                tout.append(pOut);
            
            
            warpTriangle(imagesNorm[i], img, tin, tout);


        # Add image intensities for averaging
        output = output + img;


    # Divide by numImages to get average
    output = output / numImages;
    return efr.EasyImage(output)    
    
    

if __name__ == '__main__' :
    
    Path = 'presidents/'
    
    # Dimensions of output image
    w = 600;
    h = 600;

    output = average_dir(Path, (h,w))

    # Display result
    cv2.imshow('image', output.getimg()/255.0);
    cv2.waitKey(0);
