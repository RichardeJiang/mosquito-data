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
	# print(np.max(mask))
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

def pasteWithRescaling(capImg, labImg, coordinates, posX, posY, bbox, drawFlag, rescaleChoice):
	res = np.array(labImg).astype(float)
	pt1 = (int(posY + bbox[2] - 2), int(posX + bbox[0]) - 2)
	pt2 = (int(posY + bbox[3] + 2), int(posX + bbox[1]) + 2)

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
		if rescaleChoice == 0:
			newPixel[0] = (curr[0] - minB) * 1.0 / (maxB - minB)
			newPixel[1] = (curr[1] - minG) * 1.0 / (maxG - minG)
			newPixel[2] = (curr[2] - minR) * 1.0 / (maxR - minR)
		elif rescaleChoice == 1:
			newPixel[0] = 1.0 - abs(curr[0] - maxB) * 1.0 / 255
			newPixel[1] = 1.0 - abs(curr[1] - maxG) * 1.0 / 255
			newPixel[2] = 1.0 - abs(curr[2] - maxR) * 1.0 / 255
		else:
			newPixel[0] = (curr[0] - minB) * 1.0 / 255
			newPixel[1] = (curr[1] - minG) * 1.0 / 255
			newPixel[2] = (curr[2] - minR) * 1.0 / 255
		return newPixel

	def checkXValidity(xValue):
		# return 0 if valid, 1 if x < 0, 2 if x >= 1080
		if xValue >= 0 and xValue < 1080:
			return 0
		elif xValue < 0:
			return 1
		else:
			return 2

	def checkYValidity(yValue):
		# return 0 if valid, 1 if y < 0, 2 if y >= 1920
		if yValue >= 0 and yValue < 1920:
			return 0
		elif yValue < 0:
			return 1
		else:
			return 2

	positionLabeling = [int(bbox[0] / 2 + bbox[1] / 2 + posX), int(bbox[2] / 2 + bbox[3] / 2 + posY)]
	# check of the center of the mosquito has flying out of the current frame
	xError = checkXValidity(positionLabeling[0])
	yError = checkYValidity(positionLabeling[1])
	# print("centroid: ", positionLabeling)
	# print("real pasting position: ", [posX + coordinates[0][0], posY + coordinates[0][1]])
	# all cases: 
	# 1: x ok, y < 0
	# 2: x ok, y > 1920
	# 3: x < 0, y ok
	# 4: x > 1080, y ok
	# 5: else
	if xError + yError > 0:
		if xError == 0:
			return [], positionLabeling, yError
		elif yError == 0:
			return [], positionLabeling, xError + 2
		else:
			return [], positionLabeling, 5
	# res = res.astype(float)
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


def old(capImg, labImg, coordinates, posX, posY, bbox, drawFlag, rescaleChoice):
	res = np.array(labImg).astype(float)
	pt1 = (int(posY + bbox[2] - 2), int(posX + bbox[0]) - 2)
	pt2 = (int(posY + bbox[3] + 2), int(posX + bbox[1]) + 2)

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
		if rescaleChoice == 0:
			newPixel[0] = (curr[0] - minB) * 1.0 / (maxB - minB)
			newPixel[1] = (curr[1] - minG) * 1.0 / (maxG - minG)
			newPixel[2] = (curr[2] - minR) * 1.0 / (maxR - minR)
		elif rescaleChoice == 1:
			newPixel[0] = 1.0 - abs(curr[0] - maxB) * 1.0 / 255
			newPixel[1] = 1.0 - abs(curr[1] - maxG) * 1.0 / 255
			newPixel[2] = 1.0 - abs(curr[2] - maxR) * 1.0 / 255
		else:
			newPixel[0] = (curr[0] - minB) * 1.0 / 255
			newPixel[1] = (curr[1] - minG) * 1.0 / 255
			newPixel[2] = (curr[2] - minR) * 1.0 / 255
		return newPixel

	positionLabeling = [int(bbox[0] / 2 + bbox[1] / 2 + posX), int(bbox[2] / 2 + bbox[3] / 2 + posY)]
	print("position: ", positionLabeling)
	# res = res.astype(float)
	for ele in coordinates:
		# print(ele)
		xIndex = int(posX + ele[0])
		yIndex = int(posY + ele[1])
		# res[xIndex][yIndex] *= (capImg[ele[0]][ele[1]] / 255.0)
		try:
			res[xIndex][yIndex] *= rescalePixel(capImg[ele[0]][ele[1]])
		except IndexError:
			# print(ele)
			# print(capImg[ele[0]][ele[1]])
			continue
			# print(rescalePixel(capImg[ele[0]][ele[1]]))
		
	# if drawFlag:
	# 	cv2.rectangle(res,pt1,pt2,(0,0,255),1)
	return np.uint8(res), positionLabeling

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

def angleC(p2, p1):
	ang1 = np.arctan2(*p1[::-1])
	ang2 = np.arctan2(*p2[::-1])
	return np.rad2deg((ang1 - ang2) % (2 * np.pi))

if (__name__ == "__main__"):
	bgimg = cv2.imread("images/biology.png")

	mat = scipy.io.loadmat('tracks/MVI_7521_output_multi.mat')['allSavedTracks'][0]

	# mat = [ele for ele in mat if ele[4][0][0] > 30 and ele[4][0][0] < 50] # only preserve the tracks lasts for more than 60 frames (2 seconds)
	# length of qualified mat: 9
	mat = [ele for ele in mat if ele[4][0][0] > 30]
	mat = mat[2:4]

	boundingboxes = [ele[7] for ele in mat]
	frameNum = [ele[6][0] for ele in mat]
	trackNum = len(boundingboxes)
	# boundingboxes = [ele for s in boundingboxes for ele in s]
	# centroids = [ele[:2] for ele in boundingboxes]
	# centroids = [ele[7][:2] for ele in mat]

	angles = []
	relativeAngles = []
	for i in range(len(boundingboxes) - 1):
		angle = calculateAngleBetweenTracks(boundingboxes[i], boundingboxes[i + 1])
		angles.append(angle)
		
		lastVector = boundingboxes[i][-1][:2] - boundingboxes[i][-2][:2]
		firstVector = boundingboxes[i + 1][1][:2] - boundingboxes[i + 1][0][:2]
		

	# mosquito starts flying from here; center of the bgimg
	startingPos = np.array([200, 300])
	angles = [0] + angles

	# stats:
	# bgimg: 1920 x 1080
	cap = cv2.VideoCapture('videos/MVI_7521.MOV')
	cap1 = cv2.VideoCapture('masks/MVI_7521_output_multi_BG.avi')
	# fps = cap.get(cv2.CAP_PROP_FPS)
	frameCount = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
	fps = 30

	# resWidth = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
	# resHeight = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
	resHeight, resWidth, c = bgimg.shape
	downScale = 3.0

	resultList = []
	resultList1 = []
	resultList2 = []

	standardWidth = 1920
	standardHeight = 1080

	standardLength = 5500

	pastePosX = startingPos[0]
	pastePosY = startingPos[1]

	testLength = 1200


	MShift = np.float32([[1, 0, 0], [0, 1, 0]])
	firstCentroidOnCanvas = [0, 0]
	lastCentroidOnCanvas = [0, 0]
	tempRotationMatrix = np.array([[1,0],[0,1]])

	theta = 0

	positionLabels = []

	drawRectangleFlag = False
	changeTrackShiftCompensation = [0, 0]

	mirrorFlag = False
	reenteringAllowance = 8

	labels = []

	previousEndingCentroid = np.array([0, 0]) #boundingboxes[i - 1][-1][:2]
	previousStartingCentroid = np.array([0, 0]) # boundingboxes[i - 1][0][:2]

	"""
	preprocess the bounding boxes to handle the mirror logic
	"""
	processedBoundingBoxes = []
	for i in range(trackNum):
		currentTrackCentroids = [ele for ele in boundingboxes[i]]

		"""
		comment out the following mirror code for testing
		"""
		rotationDegree = angles[i]

		"""
		Test whether we need to mirror the image
		"""
		# if rotationDegree < 90 or rotationDegree > 270:
		# 	mirrorFlag = False
		# else:
		# 	for j in range(len(currentTrackCentroids)):
		# 		currentTrackCentroids[j][0] = 1919 - currentTrackCentroids[j][0]

		processedBoundingBoxes.append(currentTrackCentroids)


	processedBoundingBoxes = np.array(processedBoundingBoxes)

	for i in range(trackNum):
		currentTrackFrames = frameNum[i]
		currentTrackCentroids = [ele[:2] for ele in processedBoundingBoxes[i]]
		assert len(currentTrackFrames) == len(currentTrackCentroids), "number of frames not matching with the number of centroids"
		firstCentroid = currentTrackCentroids[0]
		lastCentroid = currentTrackCentroids[-1]
		movementVector = np.array(lastCentroid) - np.array(firstCentroid)
		print(i)
		print("original movement vector: ", movementVector)
		print("first centroid: ", firstCentroid)
		print("last centroid: ", lastCentroid)

		rotationDegree = angles[i]
		print("calculated rotation degree: ", rotationDegree)


		"""
		get the shift distance between the two tracks
		"""

		if i > 0:
			"""
			Test whether we need to mirror the image
			"""
			if rotationDegree < 90 or rotationDegree > 270:
				mirrorFlag = False
			else:
				mirrorFlag = True


			"""
			Smoothing the connection between the two tracks by moving the shifting distance a bit based on the 2 vectors
			"""
			previousLastVector = [0, 0]
			previousLastVector[0] = processedBoundingBoxes[i - 1][-1][0] - processedBoundingBoxes[i - 1][-2][0]
			previousLastVector[1] = processedBoundingBoxes[i - 1][-1][1] - processedBoundingBoxes[i - 1][-2][1]
			# previousStepLength = np.sqrt(previousLastVector[0]**2+previousLastVector[1]**2)
			currentFirstVector = [0, 0]
			currentFirstVector[0] = processedBoundingBoxes[i][2][0] - processedBoundingBoxes[i][1][0]
			currentFirstVector[1] = processedBoundingBoxes[i][2][1] - processedBoundingBoxes[i][1][1]
			# currentStepLength = np.sqrt(currentFirstVector[0]**2+currentFirstVector[1]**2)
			# movementRatio = (currentStepLength + previousStepLength) / 2 / previousStepLength

			minorAdjustX = int((previousLastVector[0] + currentFirstVector[0]) / 2)
			minorAdjustY = int((previousLastVector[1] + currentFirstVector[1]) / 2)


			previousEndingCentroid = processedBoundingBoxes[i - 1][-1][:2]
			xShift = firstCentroid[0] - previousEndingCentroid[0] + minorAdjustX
			yShift = firstCentroid[1] - previousEndingCentroid[1] + minorAdjustY

			previousStartingCentroid = processedBoundingBoxes[i - 1][0][:2]
			previousTrackShiftX = previousEndingCentroid[0] - previousStartingCentroid[0]
			previousTrackShiftY = previousEndingCentroid[1] - previousStartingCentroid[1]
			MShift[0][2] -= xShift
			MShift[1][2] -= yShift

			firstCentroidOnCanvas[0] += previousTrackShiftX
			firstCentroidOnCanvas[1] += previousTrackShiftY


		else:
			# first track: mosquito starts flying from center of the image
			pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
			pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)


			firstCentroidOnCanvas[0] = int(standardLength /2 - standardWidth/2) + firstCentroid[0]
			firstCentroidOnCanvas[1] = int(standardLength /2 - standardHeight/2) + firstCentroid[1]


		# M = cv2.getRotationMatrix2D((firstCentroidOnCanvas[0], firstCentroidOnCanvas[1]), rotationDegree, 1)
		

		for frameCounter, frameID in enumerate(currentTrackFrames):

			cap1.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, preMask = cap1.read()
			mask = cv2.cvtColor(preMask, cv2.COLOR_BGR2GRAY)
			cap.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, img = cap.read()

			# if mirrorFlag:
			# 	img = cv2.flip(img, 1)
			# 	mask = cv2.flip(mask, 1)

			bkimg = np.array(img)
			radius = int(1.0 * max(processedBoundingBoxes[i][frameCounter][-1], processedBoundingBoxes[i][frameCounter][-2]))
			center = currentTrackCentroids[frameCounter]


			mask = clearUnrelatedMasks(mask, center, radius)


			'''
			Start of logic 1: rotate the whole mask and img by the angle, and paste it to bgimg
			'''
			# insert the img and mask into a larger 0 matrix, so later when shifting and rotating we won't lose information
			largerImg = np.zeros(shape = (standardLength, standardLength, 3))
			largerMask = np.zeros(shape = (standardLength, standardLength))

			largerImg[int(len(largerImg) /2 - standardHeight/2) : int(len(largerImg) / 2 + standardHeight/2), int(len(largerImg) /2 - standardWidth/2) : int(len(largerImg) / 2 + standardWidth/2)] = img
			largerMask[int(len(largerImg) /2 - standardHeight/2) : int(len(largerImg) / 2 + standardHeight/2), int(len(largerImg) /2 - standardWidth/2) : int(len(largerImg) / 2 + standardWidth/2)] = mask

			img = largerImg
			mask = largerMask


			try:
				dst = cv2.warpAffine(mask, MShift, (standardLength, standardLength))
				dstImg = cv2.warpAffine(img, MShift, (standardLength, standardLength))

				# dst = cv2.warpAffine(dst, M, (standardLength, standardLength))
				# dstImg = cv2.warpAffine(dstImg, M, (standardLength, standardLength))

				
			except:
				print("mosquitoes moving to the edge in raw video, jumping to the next track")
				break

			

			dst = cv2.resize(dst, None, fx = 1/downScale, fy = 1/downScale)
			dstImg = cv2.resize(dstImg, None, fx = 1/downScale, fy = 1/downScale)


			# co = getObjFromMask(dst)
			# resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)
			co, bbox = getObjFromMaskWhitish(dst)
			if len(co) == 0:
				resultList.append(bgimg)
				resultList1.append(bgimg)
				resultList2.append(bgimg)
				labels.append(0)
			else:

				# resulting, positionLabeling = pasteAfterRotationWhitish(dstImg, bgimg, co, pastePosX, pastePosY, bbox, drawRectangleFlag)
				resulting, positionLabeling, errorMsg = pasteWithRescaling(dstImg, bgimg, co, pastePosX, pastePosY, bbox, drawRectangleFlag, 0)
				resulting1, positionLabeling1, errorMsg1 = pasteWithRescaling(dstImg, bgimg, co, pastePosX, pastePosY, bbox, drawRectangleFlag, 1)
				resulting2, positionLabeling2, errorMsg2 = pasteWithRescaling(dstImg, bgimg, co, pastePosX, pastePosY, bbox, drawRectangleFlag, 2)
				
				if len(resultList) < 5:
					# print(resulting)
					# print(positionLabeling)
					# print(errorMsg)
					# print()
					cv2.imwrite('output/images/mask-' + str(len(resultList)) + '.png', dst)
					cv2.imwrite('output/images/raw-' + str(len(resultList)) + '.png', dstImg)
					cv2.imwrite('output/images/rescaleAll-' + str(len(resultList)) + '.png', resulting)
					cv2.imwrite('output/images/rescaleBrighter-' + str(len(resultList)) + '.png', resulting1)
					cv2.imwrite('output/images/rescaleDarker-' + str(len(resultList)) + '.png', resulting2)
					cv2.imwrite('output/images/clear-' + str(len(resultList)) + '.png', bkimg)

				if len(resulting) > 0:
					resultList.append(resulting)
					resultList1.append(resulting1)
					resultList2.append(resulting2)
					labels.append(1)
				else:
					print("shifting invoked!")
					print("error message: ", errorMsg)
					numOfNeg = random.randint(20,61)
					for tt in range(numOfNeg):
						resultList.append(bgimg)
						resultList1.append(bgimg)
						resultList2.append(bgimg)
						labels.append(0)
					# all cases: 
					# 1: x ok, y < 0
					# 2: x ok, y > 1920
					# 3: x < 0, y ok
					# 4: x > 1080, y ok
					# 5: else (4 corners beyond)
					if errorMsg > 4:

						# assume: re-start from the middle of the left edge
						pastePosY += (reenteringAllowance - positionLabeling[0])
						pastePosX += int(resHeight / 2 - positionLabeling[1])
					else:
						# assume: for now jump to the opposite side
						if errorMsg == 1:
							pastePosY += int(resWidth - reenteringAllowance)
							# MShift[0][2] += (6000 - 10) # 1910
						elif errorMsg == 2:
							# MShift[0][2] -= (6000 - 10)
							pastePosY -= int(resWidth - reenteringAllowance)
						elif errorMsg == 3:
							# MShift[1][2] += (standardHeight - 10) # 1070
							pastePosX += int(resHeight - reenteringAllowance)
						else:
							pastePosX -= int(resHeight - reenteringAllowance)
							# MShift[1][2] -= (standardHeight - 10)



				positionLabels.append(positionLabeling)

			# if len(resultList) < 40:
			# 	cv2.imwrite('output/images/mask-' + str(len(resultList)) + '.png', dst)
			# 	cv2.imwrite('output/images/raw-' + str(len(resultList)) + '.png', dstImg)
			# 	cv2.imwrite('output/images/composite-' + str(len(resultList)) + '.png', resulting)
			# 	cv2.imwrite('output/images/clear-' + str(len(resultList)) + '.png', bkimg)



			if len(resultList) >= testLength:
				break

		if len(resultList) >= testLength:
			break

	
	np.savetxt('output/testanotheredge.csv', np.array(labels).astype(int), fmt='%i', delimiter=',')

	cap.release()
	cap1.release()


	# fourcc = cv2.FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter('output/0402All.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)
	video1 = cv2.VideoWriter('output/0402Brighter.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)
	video2 = cv2.VideoWriter('output/0402Darker.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

	for frame in resultList:
		video.write(frame)
	for frame in resultList1:
		video1.write(frame)
	for frame in resultList2:
		video2.write(frame)


	pass