import cv2
import numpy as np
from PIL import Image

def get_sift(image1, image2):
    image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    key_pts_1, features_1 = sift.detectAndCompute(image1,None)
    key_pts_2, features_2 = sift.detectAndCompute(image2,None)
    
    return key_pts_1, key_pts_2, features_1, features_2

def get_matches(features_1, features_2, r):
    match = cv2.BFMatcher()
    matches = match.knnMatch(features_1,features_2,k=2)
    good_matches = []
    
    for m,n in matches:
        if m.distance < r*n.distance:
            good_matches.append(m)
            
    good_matches = sorted(good_matches, key = lambda x:x.distance) 
    
    return good_matches

def draw_matches(matches, key_1, key_2, image1, image2):
    draw = dict(matchColor=(0,255,255), singlePointColor=None, flags=2)
    
    img_matches = cv2.drawMatches(image1, key_1, image2, key_2, matches[0:15], None, **draw)
    
    return img_matches

def get_homography(key_1, key_2, matches):
    src_pts = np.float32([ key_1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ key_2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
    matrix, x = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    return matrix

def transform_and_stitch(image1, image2, matrix):
    w = image1.shape[1]+image2.shape[1]
    h = image1.shape[0]+image2.shape[0]
    
    stitched_image = cv2.warpPerspective(image1, matrix, (w,h))
    stitched_image[0:image2.shape[0],0:image2.shape[1]] = image2
    
    return trim(stitched_image)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])     
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame

def thumbnail(image, basewidth):
    img = Image.fromarray(image)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    
    return img