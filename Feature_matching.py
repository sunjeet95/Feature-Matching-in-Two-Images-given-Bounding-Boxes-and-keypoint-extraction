#Sunjeet Jena 7:10 PM, Saturday, 11th August 2018|
#This code is for feature mapping between two images
# By default it has been set for using test code and also by default it has been set to show the image. 
# Comment the the test code below and directly use the main code to pass image 1 (left), image 2 (right), set of bounding boxes of image 1 and set of bounding boxes of image 2
# set of bounding box means its a list of lists where each individual list corresponds to a bounding box in an image
# Each bounding box has to be in the following format [[x11,y11],[x12,y12],[x22,y22],[x21,y21]]
"""
(x11,y11)	(x12,y12)
	|---------|
	|		  |	
	|		  | 	
	|		  | 	
	|		  | 		
	|---------|
(x21,y21)	(x22,y22)
"""
import numpy as np
import cv2
from matplotlib import path
from matplotlib import pyplot as plt
import math
from itertools import chain
import time

def Get_path(set_of_bounding_box):
	# This function is to get the paths of all bounding boxes in a given set
	# 'set_of_bounding_box' is a list which contains lists of all bounding points in an image

	paths_set=[] # Emplty list to store  paths of all the the bounding boxes in a given image
	for i in set_of_bounding_box:
		p = path.Path(i)		#Get the path using matplotlib.path function
		paths_set.append(p)		#Store the path
	return paths_set

def disparity(img1,img2, bounding_box_1, bounding_box_2):
	# This function will return a set of all all matched keypoints after feature matching
	# The output of the function is a list of lists where each individual list contain two tuples with first tuple representing the keypoint coordinate in image 1 and tuple 2 is of image 2
	#[[(),()],[(),()],....]	

	# Initiate ORB detector
	#orb = cv2.ORB_create(nfeatures=100, scoreType=cv2.ORB_FAST_SCORE)
	orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, 
						firstLevel=0, nfeatures=100) #Change the edgethreshold and nfeatures for more harris corner/keypoint detection

	# Get the keypoints in both the images using ORB Detector 
	kp1 = orb.detect(img1,None)		#Getting the keypoints in image 1
	kp2 = orb.detect(img2,None)		#Getting the keypoints in image 2

	paths_1=Get_path(bounding_box_1)	# Get the paths of all the bounding boxes in image 1
	paths_2=Get_path(bounding_box_2)	# Get the paths of all the bounding boxes in image 2

	kp1={c:p for c,p  in zip ((range(len(kp1))), kp1)}	# Setting dictionary to the the keypoints of image1
	kp2={c:p for c,p  in zip ((range(len(kp2))), kp2)}	# Setting dictionary to the the keypoints of image2

	for i,j in zip(kp1, kp2):
		#This for loop is for checking if a keypoint lies inside the given set of bounding box/path or not

		check_1=[path_.contains_points([kp1[i].pt]) for path_ in paths_1] #Checking all keypoints in image1 with corresponding bounding box set of image 1
		check_2=[path_.contains_points([kp2[j].pt]) for path_ in paths_2] #Checking all keypoints in image2 with corresponding bounding box set of image 2
		
		if True not in check_1:
			del kp1[i]	#Deleting the keypoint if not present in any of the bounding box
		if True not in check_2:
			del kp2[j]	#Deleting the keypoint if not present in any of the bounding box

	kp1=kp1.values() 	#Getting all the filtered keypoints
	kp2=kp2.values()	#Getting all the filtered keypoints

	kp1, des1 = orb.compute(img1,kp1)	#Computing descriptors for image 1
	kp2, des2 = orb.compute(img2,kp2)	#Computing descriptors for image 2

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	#bf = cv2.FlannBasedMatcher()
	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)


	# Comment the next two lines for disabling displaying of matched features of the image
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
	plt.imshow(img3),plt.show()

	list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] #Get The corresponding keypoing matches for image 1
	list_kp2 = [kp2[mat.trainIdx].pt for mat in matches] #Get The corresponding keypoing matches for image 2

	matched_key_list=[]
	for i,j in zip (list_kp1, list_kp2):
		matched_key_list.append([i,j])
	
	return matched_key_list



def main(img1,img2, bounding_box_1, bounding_box_2):
	#Sunjeet Jena 1:57 AM, Wednesday, 15th August 2018|
	#This main function where the two images, Image 1 (Left Image) and Image 2 (Right Image) are iven as input
	# 'bounding_box_1' is a list which contains the lists of bounding boxes of Image 1
	# 'bounding_box_2' is a list which contains the lists of bounding boxes of Image 2

	list_key_matches_=disparity(img1, img2, bounding_box_1, bounding_box_2)	#Get all the matched keypoint coordinates

	return list_key_matches_

####
#Test Code
#Currently the bounding box of left most cone is only given
img1 = cv2.imread('tsukuba-l.png') # Left Image
img2 = cv2.imread('tsukuba-r.png') # Right Image
time_1=time.time()
bounding_box_1=[[(100,135), (205, 135),(205,260),(100,260)]]
bounding_box_2=[[(100,135), (205, 135),(205,260),(100,260)]]
####


list_key_matches=main(img1, img2, bounding_box_1, bounding_box_2) #Get all the matched keypoint coordinates
print time.time()-time_1
print list_key_matches