import cv2
# import cv2.cv as cv
import numpy as np
import os
import random
from vidstab import VidStab

if (__name__ == "__main__"):
	
	videoNames = ['MVI_7515', 'MVI_7516', 'MVI_7517', 'MVI_7520', 'MVI_7521', 'MVI_7525', 'MVI_7572', 'MVI_7573', 'MVI_7580', 'MVI_7581', 'MVI_7582']
	videoNames = ['MVI_7521', 'MVI_7572', 'MVI_7573', 'MVI_7580', 'MVI_7581', 'MVI_7582']
	maskNames = ['masks/' + ele + '_output_multi_BG.avi' for ele in videoNames]
	trackNames = ['tracks/' + ele + '_output_multi.mat' for ele in videoNames]
	videoNames = ['videos/' + ele + '.MOV' for ele in videoNames]

	# print(mixed)
	bgVideoList = os.listdir('bg/tmp/drinking')
	bgVideoList = ['bg/tmp/drinking/' + ele for ele in bgVideoList if '.mp4' in ele]
	bgVideoList = bgVideoList[5:170]
	# bgVideoList = ['bg/tmp/drinking/a001-0855C.mp4']

	# 6*30 = 180 frames
	bgVideoIndex = 0
	while bgVideoIndex < len(bgVideoList):
	# for bgVideo in bgVideoList:
		stabilizer = VidStab()
		if bgVideoIndex > 0 and capBG != None:
			capBG.release()

		resultList = []
		labels = []
		bgVideo = bgVideoList[bgVideoIndex]
		bgVideoIndex += 1
		resultTitle = bgVideo.split('/')[3].split('.')[0]
		print(resultTitle)
		# resultName = bgVideo.split('.')[0] + '.avi'
		resultName = bgVideo.replace('.mp4', '.avi')
		capBG = cv2.VideoCapture(bgVideo)
		print(resultName)

		_, testImg = capBG.read()

		resHeight, resWidth, c = testImg.shape
		print("height: ", resHeight)
		print("width: ", resWidth)
		# print(resHeight)
		startingPos = np.array([200, 1800])
		downScale = 6.0
		if resHeight == 1080 and resWidth == 1920:
			downScale = 5.0
		elif resHeight == 720:
			downScale = 6.0
			startingPos[1] = 1100
		else:
			# bgVideoIndex -= 1
			capBG.release()
			continue

		if int(capBG.get(cv2.CAP_PROP_FPS)) > 30:
			capBG.release()
			continue

		stabilizer.stabilize(input_path = bgVideo, output_path = resultName)