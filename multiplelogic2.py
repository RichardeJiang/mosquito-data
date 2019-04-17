import cv2
# import cv2.cv as cv
import numpy as np
import scipy.io
import os
import random

"""
TODO:
current pasting logic: for each pixel, multiplying the mosquito pixel with the background pixel
maybe use the more precise logic: for the center body: replace; while for the whitish surroundings, use multiplication
Note: here the bodyColorThreshold = 16 is predefined by the previous tryer
Note 2: Nitin's old logic is wrong, where im2uint8 is used: it'll scale the whole image
"""

def getObjFromMaskWhitish(mask, bodyColorThreshold = 4):
	blurred = []
	body = []
	coordinates = []
	# checkingThreshold = mask > bodyColorThreshold
	checkingThreshold = np.argwhere(mask > bodyColorThreshold)
	try:
		minRow = np.min(checkingThreshold[:, :1])
		maxRow = np.max(checkingThreshold[:, :1])
		minCol = np.min(checkingThreshold[:, 1:])
		maxCol = np.max(checkingThreshold[:, 1:])
		bbox = [minRow, maxRow, minCol, maxCol]
		coordinates = checkingThreshold

		# below: originally looping through to get the coordinates
		# print(coordinates.shape)
		# print(coordinates)
		# for row in range(len(mask)):
		# 	for col in range(len(mask[0])):
		# 		if mask[row][col] >= bodyColorThreshold:
		# 			coordinates.append((row, col))

		return coordinates, bbox
	except ValueError:
		print("Encountering zero qualified mask: ")
		# print(mask)
		print(type(mask))
		print(mask.shape)
		print("largest value in mask: ", np.max(mask))
		return [], []

def pasteAfterRotationWhitish(capImg, labImg, coordinates, posX, posY, bbox, drawFlag):
	res = np.array(labImg).astype(float)
	pt1 = (int(posY + bbox[2] - 2), int(posX + bbox[0]) - 2)
	pt2 = (int(posY + bbox[3] + 2), int(posX + bbox[1]) + 2)

	positionLabeling = [int(bbox[0] / 2 + bbox[1] / 2 + posX), int(bbox[2] / 2 + bbox[3] / 2 + posY)]
	# res = res.astype(float)
	for ele in coordinates:
		# print(ele)
		xIndex = int(posX + ele[0])
		yIndex = int(posY + ele[1])
		res[xIndex][yIndex] *= (capImg[ele[0]][ele[1]] / 255.0)
		# if printFlag:
		# 	print (xIndex, yIndex)
	# print(np.max(res))
	if drawFlag:
		cv2.rectangle(res,pt1,pt2,(0,0,255),1)
	return np.uint8(res), positionLabeling

def pasteWithRescaling(capImg, labImg, coordinates, posX, posY, bbox, drawFlag, bgWidth, bgHeight):
	res = np.array(labImg).astype(float)
	

	minB = capImg[coordinates[0][0]][coordinates[0][1]][0]
	maxB = capImg[coordinates[0][0]][coordinates[0][1]][0]
	minG = capImg[coordinates[0][0]][coordinates[0][1]][1]
	maxG = capImg[coordinates[0][0]][coordinates[0][1]][1]
	minR = capImg[coordinates[0][0]][coordinates[0][1]][2]
	maxR = capImg[coordinates[0][0]][coordinates[0][1]][2]
	for ele in coordinates:
		if capImg[ele[0]][ele[1]][0] < minB:
			minB = capImg[ele[0]][ele[1]][0]
		elif capImg[ele[0]][ele[1]][0] > maxB:
			maxB = capImg[ele[0]][ele[1]][0]

		if capImg[ele[0]][ele[1]][1] < minG:
			minG = capImg[ele[0]][ele[1]][1]
		elif capImg[ele[0]][ele[1]][1] > maxG:
			maxG = capImg[ele[0]][ele[1]][1]

		if capImg[ele[0]][ele[1]][2] < minR:
			minR = capImg[ele[0]][ele[1]][2]
		elif capImg[ele[0]][ele[1]][2] > maxR:
			maxR = capImg[ele[0]][ele[1]][2]

	def rescalePixel(curr):
		newPixel = np.array([0.0, 0.0, 0.0])
		newPixel[0] = (curr[0] - minB) * 1.0 / (maxB - minB)
		newPixel[1] = (curr[1] - minG) * 1.0 / (maxG - minG)
		newPixel[2] = (curr[2] - minR) * 1.0 / (maxR - minR)
		# newPixel[0] = 1.0 - abs(curr[0] - maxB) * 1.0 / 255
		# newPixel[1] = 1.0 - abs(curr[1] - maxG) * 1.0 / 255
		# newPixel[2] = 1.0 - abs(curr[2] - maxR) * 1.0 / 255
		# newPixel[0] = (curr[0] - minB) * 1.0 / 255
		# newPixel[1] = (curr[1] - minG) * 1.0 / 255
		# newPixel[2] = (curr[2] - minR) * 1.0 / 255
		return newPixel

	def checkXValidity(xValue):
		# return 0 if valid, 1 if x < 0, 2 if x >= 1080
		if xValue >= 0 and xValue < bgHeight:
			return 0
		elif xValue < 0:
			return 1
		else:
			return 2

	def checkYValidity(yValue):
		# return 0 if valid, 1 if y < 0, 2 if y >= 1920
		if yValue >= 0 and yValue < bgWidth:
			return 0
		elif yValue < 0:
			return 1
		else:
			return 2

	patchHeight = len(capImg)
	patchWidth = len(capImg[0])
	posX -= int(patchHeight / 2)
	posY -= int(patchWidth / 2)
	pt1 = (int(posY + bbox[2] - 2), int(posX + bbox[0]) - 2)
	pt2 = (int(posY + bbox[3] + 2), int(posX + bbox[1]) + 2)
	positionLabeling = [int(bbox[0] / 2 + bbox[1] / 2 + posX), int(bbox[2] / 2 + bbox[3] / 2 + posY)]
	# check of the center of the mosquito has flying out of the current frame
	xError = checkXValidity(positionLabeling[0])
	yError = checkYValidity(positionLabeling[1])
	print(positionLabeling)
	# print("centroid: ", positionLabeling)
	# print("real pasting position: ", [posX + coordinates[0][0], posY + coordinates[0][1]])

	# !!! all beyond-boundary cases: 
	# 1: x ok, y < 0
	# 2: x ok, y > 1920
	# 3: x < 0, y ok
	# 4: x > 1080, y ok
	# 5: else (4 corners and beyond)
	if xError + yError > 0:
		if xError == 0:
			return [], positionLabeling, yError
		elif yError == 0:
			return [], positionLabeling, xError + 2
		else:
			return [], positionLabeling, 5

	
	for ele in coordinates:
		# print(ele)
		xIndex = int(posX + ele[0])
		yIndex = int(posY + ele[1])
		xError = checkXValidity(xIndex)
		yError = checkYValidity(yIndex)
		if xError + yError == 0:
			# res[xIndex][yIndex] *= (capImg[ele[0]][ele[1]] / 255.0)
			res[xIndex][yIndex] *= rescalePixel(capImg[ele[0]][ele[1]])
		else:
			continue
			
		
	if drawFlag:
		cv2.rectangle(res,pt1,pt2,(0,0,255),1)
	return np.uint8(res), positionLabeling, 0


def clearUnrelatedMasks(mask, centroid, radius):
	res = np.zeros(shape = mask.shape)
	radius = int(radius * 1.5)

	startingX = centroid[1] - radius
	endingX = centroid[1] + radius
	startingY = centroid[0] - radius
	endingY = centroid[0] + radius

	startingX = 0 if startingX < 0 else startingX
	startingY = 0 if startingY < 0 else startingY
	endingX = len(res) - 1 if endingX >= len(res) else endingX
	endingY = len(res[0]) - 1 if endingY >= len(res[0]) else endingY
	# print(mask)
	# print(centroid)
	res[startingX : endingX, startingY : endingY] = mask[startingX : endingX, startingY : endingY]

	return res

def getLocalPatch(mask, centroid, radius):
	res = np.zeros(shape = mask.shape)
	radius = int(radius * 1.5)

	startingX = centroid[1] - radius
	endingX = centroid[1] + radius
	startingY = centroid[0] - radius
	endingY = centroid[0] + radius

	startingX = 0 if startingX < 0 else startingX
	startingY = 0 if startingY < 0 else startingY
	endingX = len(res) - 1 if endingX >= len(res) else endingX
	endingY = len(res[0]) - 1 if endingY >= len(res[0]) else endingY

	return mask[startingX : endingX, startingY : endingY]

# track1 and track2 are 2 list of arrays
def calculateAngleBetweenTracks(track1, track2):
	lastVector = track1[-1][:2] - track1[-2][:2]
	firstVector = track2[1][:2] - track2[0][:2]

	# source: https://stackoverflow.com/a/31735642/3455398 |  counter-clockwise angle between the two
	# e.g. a=[1,0] b=[0,1] => angle_between(a,b) returns 90
	# note: counter clockwise on normal coordinate; clockwise on image coordinates (1920, 1080)
	def angle_between(p2, p1):
		ang1 = np.arctan2(*p1[::-1])
		ang2 = np.arctan2(*p2[::-1])
		return np.rad2deg((ang1 - ang2) % (2 * np.pi))

	# first, then last => we are trying to rotate the later one to match with the first one
	return angle_between(firstVector, lastVector)

def write3DArrayTranspose(arrayToWrite, fileName):
	tf = open(fileName, "w")
	
	for j in range(len(arrayToWrite[0])):
		for i in range(len(arrayToWrite)):
			tf.write("%s" % arrayToWrite[i][j])
			if i < len(arrayToWrite) - 1:
				tf.write(",")
			else:
				tf.write("\n")

	tf.close()
	return

if (__name__ == "__main__"):
	videoNames = ['MVI_7515', 'MVI_7516', 'MVI_7517', 'MVI_7520', 'MVI_7521', 'MVI_7525', 'MVI_7572', 'MVI_7573', 'MVI_7580', 'MVI_7581', 'MVI_7582']
	videoNames = ['MVI_7521', 'MVI_7572', 'MVI_7573', 'MVI_7580', 'MVI_7581', 'MVI_7582']
	maskNames = ['masks/' + ele + '_output_multi_BG.avi' for ele in videoNames]
	trackNames = ['tracks/' + ele + '_output_multi.mat' for ele in videoNames]
	videoNames = ['videos/' + ele + '.MOV' for ele in videoNames]

	fps = 30

	mats = [scipy.io.loadmat(ele)['allSavedTracks'][0] for ele in trackNames]
	processedMats = []
	for mat in mats:
		mat = [ele for ele in mat if ele[4][0][0] > 30]
		processedMats.append(mat)

	mixed = list(zip(videoNames, maskNames, processedMats))
	# print(mixed)
	bgVideoList = os.listdir('bg/tmp/drinking')
	bgVideoList = ['bg/tmp/drinking/' + ele for ele in bgVideoList if '.avi' in ele]
	bgVideoList = bgVideoList[5:10]
	# bgVideoList = ['bg/tmp/drinking/a001-0855C.mp4']

	# 6*30 = 180 frames
	bgVideoIndex = 0
	while bgVideoIndex < len(bgVideoList):
	# for bgVideo in bgVideoList:
		if bgVideoIndex > 0 and cap != None:
			cap.release()
		if bgVideoIndex > 0 and cap1 != None:
			cap1.release()
		if bgVideoIndex > 0 and capBG != None:
			capBG.release()

		resultList = []
		labels = []
		bgVideo = bgVideoList[bgVideoIndex]
		bgVideoIndex += 1
		resultTitle = bgVideo.split('/')[3].split('.')[0]
		print(resultTitle)
		capBG = cv2.VideoCapture(bgVideo)
		print("bg fps: ", capBG.get(cv2.CAP_PROP_FPS))
		

		_, testImg = capBG.read()

		resHeight, resWidth, c = testImg.shape
		print("height: ", resHeight)
		print("width: ", resWidth)
		# print(resHeight)
		
		downScale = 6.0
		if resHeight == 1080 and resWidth == 1920:
			downScale = 5.0
		elif resHeight == 720:
			downScale = 6.0
			# startingPos[1] = 1100
		else:
			# bgVideoIndex -= 1
			capBG.release()
			continue

		if int(capBG.get(cv2.CAP_PROP_FPS)) > 30:
			capBG.release()
			continue


		"""
		All checkings have been finished: the video are okay to use, proceed
		"""
		
		# get the number of mosquitoes to be synthesized in the current output
		currOutputNumOfMos = random.randint(1,10)
		print("Current output mosquito number: ", currOutputNumOfMos)

		positionLabels = [[] for n in range(currOutputNumOfMos)]
		labels = [[] for n in range(currOutputNumOfMos)]

		# the track is okay, now retrieve all frames from the current bg video
		resultList = []
		bgFrameCount = int(capBG.get(cv2.CAP_PROP_FRAME_COUNT))
		testLength = bgFrameCount - 3

		print("testLength: ", testLength)
		print("bg frame count: ", bgFrameCount)

		for i in range(testLength):
			_, bgimg = capBG.read()
			resultList.append(bgimg)

		capBG.release()

		# iterating through different mosquitoes
		mosquitoIndex = 0
		while mosquitoIndex < currOutputNumOfMos:

			# retrieve the tracks information, and check whether it's sufficient
			(videoName, maskName, mat) = random.choice(mixed)
			boundingboxes = [ele[7] for ele in mat]
			frameNum = [ele[6][0] for ele in mat]
			trackNum = len(boundingboxes)
			
			templengthList = [len(ele) for ele in frameNum]
			totalTrackFrames = sum(templengthList)

			print("total frames in tracks: ", totalTrackFrames)
			if totalTrackFrames < 60:
				# bgVideoIndex -= 1
				mosquitoIndex -= 1
				continue

			# random the starting position of the current mosquito
			startingRow = random.randint(100, resHeight - 100)
			startingCol = random.randint(100, resWidth - 100)
			startingPos = np.array([startingRow, startingCol])

			# adding the following line to test edge reentering
			# startingPos = np.array([200, 1880]) if mosquitoIndex == 0 else np.array([20, 400])

			angles = []
			for i in range(len(boundingboxes) - 1):
				angle = calculateAngleBetweenTracks(boundingboxes[i], boundingboxes[i + 1])
				angles.append(angle)
		
			angles = [0] + angles

			# the angles are read in, and we can start formulating the placeholders for the current mosquito

			# note: read in the new track video and mask;
			# cannot use the same logic as bg video, as bg video is reused for multiple tracks, 
			# but for track we use a different track every time
			cap = cv2.VideoCapture(videoName)
			cap1 = cv2.VideoCapture(maskName)
			# fps = cap.get(cv2.CAP_PROP_FPS)
			frameCount = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), cap1.get(cv2.CAP_PROP_FRAME_COUNT)))		
			currPositionList = positionLabels[mosquitoIndex]
			currLabels = labels[mosquitoIndex]

			standardWidth = 1920
			standardHeight = 1080
			fixedDownScale = 6.0

			pastePosX = startingPos[0]
			pastePosY = startingPos[1]

			# start from the first bg frame
			bgFrameCounter = 0

			drawRectangleFlag = False

			mirrorFlag = False

			previousEndingCentroid = np.array([0, 0]) #boundingboxes[i - 1][-1][:2]
			previousStartingCentroid = np.array([0, 0]) # boundingboxes[i - 1][0][:2]
			reenteringAllowance = 6

			"""
			preprocess the bounding boxes to handle the mirror logic
			"""
			processedBoundingBoxes = []
			for i in range(trackNum):
				currentTrackCentroids = [ele for ele in boundingboxes[i]]
				rotationDegree = angles[i]

				processedBoundingBoxes.append(currentTrackCentroids)


			processedBoundingBoxes = np.array(processedBoundingBoxes)

			for i in range(trackNum):
				currentTrackFrames = frameNum[i]
				downScaleList = [fixedDownScale] * len(currentTrackFrames)
				"""
				random downScale logic:
				can be from [5.0, 6.0, 7.0, 8.0]
				every 5 frames, draw a random number from [0,1,2]
				0 means downScale -= 1, 1 means no change, 2 means downScale += 1
				"""
				initialDownScale = 6.0
				downScaleCounter = 0
				for k in range(5, len(downScaleList) - 5):
					if downScaleCounter % 5 == 0:
						lucky = random.randint(0, 2)
						if lucky == 0 and initialDownScale > 5:
							initialDownScale -= 1
						elif lucky == 2 and initialDownScale < 8:
							initialDownScale += 1

						# downScaleCounter

					# else:
					downScaleCounter += 1
					downScaleList[k] = initialDownScale
				print(currentTrackFrames)
				currentTrackCentroids = [ele[:2] for ele in processedBoundingBoxes[i]]
				assert len(currentTrackFrames) == len(currentTrackCentroids), "number of frames not matching with the number of centroids"
				firstCentroid = currentTrackCentroids[0]
				lastCentroid = currentTrackCentroids[-1]
				movementVector = np.array(lastCentroid) - np.array(firstCentroid)

				rotationDegree = angles[i]
				# print("calculated rotation degree: ", rotationDegree)

				currTrackPastingPos = []

				if i == 0:
					pastingPosition = list(startingPos)
					currTrackPastingPos.append(pastingPosition)
				else:

					previousLastVector = [0, 0]
					previousLastVector[0] = processedBoundingBoxes[i - 1][-1][0] - processedBoundingBoxes[i - 1][-2][0]
					previousLastVector[1] = processedBoundingBoxes[i - 1][-1][1] - processedBoundingBoxes[i - 1][-2][1]

					currentFirstVector = [0, 0]
					currentFirstVector[0] = processedBoundingBoxes[i][2][0] - processedBoundingBoxes[i][1][0]
					currentFirstVector[1] = processedBoundingBoxes[i][2][1] - processedBoundingBoxes[i][1][1]

					minorAdjustX = int((previousLastVector[0] + currentFirstVector[0]) / 2)
					minorAdjustY = int((previousLastVector[1] + currentFirstVector[1]) / 2)
					
					lastPos[1] += int(minorAdjustX / fixedDownScale)
					lastPos[0] += int(minorAdjustY / fixedDownScale)

					currTrackPastingPos.append(lastPos)
					print("computed last position: ", lastPos)

					# temp = currentTrackCentroids[1:]
				for j in range(len(currentTrackCentroids) - 1):
					currDownScale = downScaleList[j + 1]
					step = currentTrackCentroids[j + 1] - currentTrackCentroids[j]
					prev = list(currTrackPastingPos[-1])
					# print(step)
					# print(type(step))
					step = step.astype(float)
					step /= currDownScale
					prev[0] += int(step[1])
					prev[1] += int(step[0])
					currTrackPastingPos.append(prev)

				lastPos = currTrackPastingPos[-1]
				print("original last position: ", lastPos)
				print("whole pasting list: ", currTrackPastingPos)


				for frameCounter, frameID in enumerate(currentTrackFrames):

					currFramePastingPos = currTrackPastingPos[frameCounter]
					currDownScale = downScaleList[frameCounter]

					cap1.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
					_, preMask = cap1.read()
					mask = cv2.cvtColor(preMask, cv2.COLOR_BGR2GRAY)
					cap.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
					_, img = cap.read()


					bkimg = np.array(img)
					radius = int(1.0 * max(processedBoundingBoxes[i][frameCounter][-1], processedBoundingBoxes[i][frameCounter][-2]))
					center = currentTrackCentroids[frameCounter]


					mask = clearUnrelatedMasks(mask, center, radius)
					localPatchMask = getLocalPatch(mask, center, 100)
					localPatchImg = getLocalPatch(img, center, 100)

					dst = cv2.resize(localPatchMask, None, fx = 1/currDownScale, fy = 1/currDownScale)
					dstImg = cv2.resize(localPatchImg, None, fx = 1/currDownScale, fy = 1/currDownScale)

					# _, bgimg = capBG.read()
					bgimg = resultList[bgFrameCounter]

					# bgFrameCounter += 1
					co, bbox = getObjFromMaskWhitish(dst)
					if len(co) == 0:
						# resultList.append(bgimg)
						print("Weird: no mosquito can be found from the mask; skipping.")
						bgFrameCounter += 1
						currLabels.append(0)
						currPositionList.append([-1, -1])
					else:
						resulting, positionLabeling, errorMsg = pasteWithRescaling(dstImg, bgimg, co, currFramePastingPos[0], currFramePastingPos[1], bbox, drawRectangleFlag, resWidth, resHeight)
				
						if len(resulting) > 0:
							# resultList.append(resulting)
							resultList[bgFrameCounter] = resulting
							bgFrameCounter += 1
							currLabels.append(1)
							currPositionList.append(positionLabeling)
						else:
							print("shifting invoked!")
							print("error message: ", errorMsg)
							numOfNeg = random.randint(20,60)
							for tt in range(numOfNeg):
								# bgimg = resultList[bgFrameCounter]
								# resultList.append(bgimg)
								bgFrameCounter += 1
								
								currLabels.append(0)
								currPositionList.append([-1, -1])
								if bgFrameCounter > testLength - 1:
									break
							# all cases: 
							# 1: x ok, y < 0
							# 2: x ok, y > 1920
							# 3: x < 0, y ok
							# 4: x > 1080, y ok
							# 5: else (4 corners and beyond)
							if errorMsg > 4:

								# assume: re-start from the middle of the left edge
								for currFrameID in range(frameCounter, len(currTrackPastingPos)):
									currTrackPastingPos[currFrameID][1] += int(reenteringAllowance - positionLabeling[0])
									currTrackPastingPos[currFrameID][0] += int(resHeight / 2 - positionLabeling[1])
								pastePosY += int(reenteringAllowance - positionLabeling[0])
								pastePosX += int(resHeight / 2 - positionLabeling[1])
								lastPos[0] += int(resHeight / 2 - positionLabeling[1])
								lastPos[1] += int(reenteringAllowance - positionLabeling[0])
							else:
								
								reenterChecker = random.randint(0, 3)
								reenterCol = random.randint(200, resWidth - 200)
								reenterRow = random.randint(200, resHeight - 200)

								# print("col: ", reenterCol)
								# print("row: ", reenterRow)
								# print("position labeling: ", positionLabeling)

								# reenterChecker = 1

								# reenter from top edge
								if reenterChecker == 0:
									print("reentering from top edge")
									for currFrameID in range(frameCounter, len(currTrackPastingPos)):
										currTrackPastingPos[currFrameID][1] += int(reenterCol - positionLabeling[1])
										currTrackPastingPos[currFrameID][0] += int(reenteringAllowance - positionLabeling[0])

									lastPos[0] += int(reenteringAllowance - positionLabeling[0])
									lastPos[1] += int(reenterCol - positionLabeling[1])

								# reenter from bottom edge
								elif reenterChecker == 1:
									print("reentering from bottom edge")
									for currFrameID in range(frameCounter, len(currTrackPastingPos)):
										currTrackPastingPos[currFrameID][1] += int(reenterCol - positionLabeling[1])
										currTrackPastingPos[currFrameID][0] += int(resHeight - reenteringAllowance - positionLabeling[0])
										# print("recalculating pasting position: ", currTrackPastingPos[currFrameID])

									lastPos[0] += int(resHeight - reenteringAllowance - positionLabeling[0])
									lastPos[1] += int(reenterCol - positionLabeling[1])

								# reenter from left edge
								elif reenterChecker == 2:
									print("reentering from left edge")
									for currFrameID in range(frameCounter, len(currTrackPastingPos)):
										currTrackPastingPos[currFrameID][1] += int(reenteringAllowance - positionLabeling[1])
										currTrackPastingPos[currFrameID][0] += int(reenterRow - positionLabeling[0])

									lastPos[0] += int(reenterRow - positionLabeling[0])
									lastPos[1] += int(reenteringAllowance - positionLabeling[1])

								# reenter from right edge
								else:
									print("reentering from right edge")
									for currFrameID in range(frameCounter, len(currTrackPastingPos)):
										currTrackPastingPos[currFrameID][1] += int(resWidth - reenteringAllowance - positionLabeling[1])
										currTrackPastingPos[currFrameID][0] += int(reenterRow - positionLabeling[0])

									lastPos[0] += int(reenterRow - positionLabeling[0])
									lastPos[1] += int(resWidth - reenteringAllowance - positionLabeling[1])

								
								# below logic: assume jumping to the opposite side
								# if errorMsg == 1:
								# 	for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								# 		currTrackPastingPos[currFrameID][1] += int(resWidth - reenteringAllowance)
								# 	pastePosY += int(resWidth - reenteringAllowance)
								# 	lastPos[1] += int(resWidth - reenteringAllowance)
								# 	# MShift[0][2] += (6000 - 10) # 1910
								# elif errorMsg == 2:
								# 	for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								# 		currTrackPastingPos[currFrameID][1] -= int(resWidth - reenteringAllowance)
								# 	pastePosY -= int(resWidth - reenteringAllowance)
								# 	lastPos[1] -= int(resWidth - reenteringAllowance)
								# elif errorMsg == 3:
								# 	# MShift[1][2] += (standardHeight - 10) # 1070
								# 	for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								# 		currTrackPastingPos[currFrameID][0] += int(resHeight - reenteringAllowance)
								# 	pastePosX += int(resHeight - reenteringAllowance)
								# 	lastPos[0] += int(resHeight - reenteringAllowance)
								# else:
								# 	for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								# 		currTrackPastingPos[currFrameID][0] -= int(resHeight - reenteringAllowance)
								# 	pastePosX -= int(resHeight - reenteringAllowance)
								# 	lastPos[0] -= int(resHeight - reenteringAllowance)


					if bgFrameCounter > testLength - 1:
						break

				if bgFrameCounter > testLength - 1:
					break

				# end of tracks loop


			cap.release()
			cap1.release()
			mosquitoIndex += 1
			print("current track finished, changing to another mosquito!")

			# end of mosquito index loop
		

		labels = np.transpose(np.array(labels))
		# positionLabels = np.transpose(np.array(positionLabels))

		print("Sanity check: num of frames in bg video: ", len(resultList))
		print("Sanity check: length of labels: ", len(labels))
		print("Sanity check: length of position labels: ", len(positionLabels[0]))

		# self-defining function to write position labels, as it is 3D array and np savetxt will complain
		np.savetxt('bg/tmp/generate/0416/labels/' + resultTitle + '.csv', np.array(labels).astype(int), fmt='%i', delimiter=',')
		write3DArrayTranspose(positionLabels, 'bg/tmp/generate/0416/labels/' + resultTitle + '-pos.csv')
		# np.savetxt('bg/tmp/generate/0416/labels/' + resultTitle + '-pos.csv', np.array(positionLabels).astype(int), fmt='%i', delimiter=',')
	
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		video = cv2.VideoWriter('bg/tmp/generate/0416/videos/logic2-' + resultTitle + '.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

		for frame in resultList:
			try:
				video.write(frame)
			except Exception:
				print(frame)
				continue

		# end of bg video loop

	pass