# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 16:48:04 2016

@author: rakhunzy
"""

import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt


# In[]

def middle_point(lst):
    return lst[len(lst)/2]

def point_index(contour, point):
    return np.argwhere(np.all(contour == point,axis=1))[0,0]

def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1 = int(start[0])
    y1 = int(start[1])
    x2 = int(end[0])
    y2 = int(end[1])
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return np.array(points)

# In[]
def read_contours(fname):
    content = []
    for s in open(fname).readlines():
        s = s.strip('\t\n').split('\t')
        image_filename = s[0]
        rect = np.array([np.float(i) for i in s[1:5]])
        coordinates = np.array([np.float(i) for i in s[5:]])
        coordinates = np.reshape(coordinates, (len(coordinates)/2, 2))
        content.append([image_filename, rect, coordinates,])
    return content

# In[]

def get_contour_section(contour, index1, index2):
    if index1 < index2:
        return contour[index1:index2]
    else:
        return np.vstack((contour[index1:],contour[:index2]))

def make_dense_contour(contour):
    contour = np.append(contour, [contour[0]], axis=0)
    new_contour = []
    for i in range(len(contour)-1):
        if i == 0:
            new_contour = get_line(contour[i],contour[i+1])
        else:
            new_contour = np.vstack((new_contour, get_line(contour[i],contour[i+1])))
    return new_contour

def double_landmarks(contour, lm_indices):
    primary_points = np.array(lm_indices)
    primary_points = np.append(primary_points, primary_points[0])
    
    new_points = []
    for i in range(len(primary_points) - 1):
        point = middle_point(get_contour_section(contour, primary_points[i], primary_points[i + 1]))
        point_idx = point_index(contour, point)
        new_points.append(point_idx)
        
    old_points = np.array([lm_indices])
    new_points = np.array([new_points])
    result = np.vstack((old_points, new_points)).transpose().reshape((1,len(lm_indices)*2))
    return np.ravel(result)

def contour_to_landmarks(contour):
    contour_max = np.max(contour, axis=0)
    contour_min = np.min(contour, axis=0)    

    point_top = middle_point(contour[np.where( contour[:,1] == contour_min[1])])
    point_bottom = middle_point(contour[np.where( contour[:,1] == contour_max[1])])
#    point_left = middle_point(contour[np.where( contour[:,0] == contour_min[0])])
#    point_right = middle_point(contour[np.where( contour[:,0] == contour_max[0])])
    lm_2 =  [point_index(contour, p) for p in (point_top, point_bottom)]
    lm_4 = double_landmarks(contour, lm_2)
    #lm_4 =  [point_index(contour, p) for p in (point_top, point_right, point_bottom, point_left)]
    lm_8 = double_landmarks(contour, lm_4)
    lm_16 = double_landmarks(contour, lm_8)
   
    return contour[lm_16]

# x y width height
def bounding_box(iterable):
    min_x, min_y = np.min(iterable, axis=0)
    max_x, max_y = np.max(iterable, axis=0)
    return np.array([min_x, min_y, max_x-min_x, max_y - min_y])

#rect = contours[0][1]
#contour = contours[0][2]

def line_to_landmark(rect, contour):
    dcontour = make_dense_contour(contour)
    landmarks = contour_to_landmarks(dcontour)
    landmarks = np.array([[lm[0] + rect[0], lm[1] + rect[1]] for lm in landmarks])
    bbox = bounding_box(landmarks)
    flat_landmarks = np.ravel(landmarks.reshape((1,len(landmarks)*2)))
    result = np.hstack((bbox, flat_landmarks))
    return result 

if False:
    input_file_name = '../../data/train/1.pts'
# In[]
input_file_name = sys.argv[1] 
output_file_name = input_file_name[:-4] + '_boxede16.lms'
contours = read_contours(input_file_name)

# In[]    
df = pd.DataFrame([c[0] for c in contours])

images = np.array([[c[0]] for c in contours])
landmark = np.array([line_to_landmark(c[1], c[2]) for c in contours])
df = pd.DataFrame(np.hstack((images, landmark)))

df.to_csv(output_file_name, header=False, index=False, sep='\t')

#plt.figure()
#plt.plot(dcontour[:,0],dcontour[:,1])    
#plt.plot(landmarks[:,0], landmarks[:,1],'ro')   

