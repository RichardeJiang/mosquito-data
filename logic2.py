import cv2
# import cv2.cv as cv
import numpy as np
import scipy.io
import os


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

def pasteWithRescaling(capImg, labImg, coordinates, posX, posY, bbox, drawFlag):
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
		newPixel[0] = (curr[0] - minB) * 1.0 / (maxB - minB)
		newPixel[1] = (curr[1] - minG) * 1.0 / (maxG - minG)
		newPixel[2] = (curr[2] - minR) * 1.0 / (maxR - minR)
		# newPixel[0] = (curr[0] - minB) * 1.0 / 255
		# newPixel[1] = (curr[1] - minG) * 1.0 / 255
		# newPixel[2] = (curr[2] - minR) * 1.0 / 255
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


def old(capImg, labImg, coordinates, posX, posY, bbox, drawFlag):
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
		newPixel[0] = (curr[0] - minB) * 1.0 / (maxB - minB)
		newPixel[1] = (curr[1] - minG) * 1.0 / (maxG - minG)
		newPixel[2] = (curr[2] - minR) * 1.0 / (maxR - minR)
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
	# res[startingX : endingX, startingY : endingY] = mask[startingX : endingX, startingY : endingY]

	# return res

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

if (__name__ == "__main__"):
	bgimg = cv2.imread("images/biology.png")

	mat = scipy.io.loadmat('tracks/MVI_7521_output_multi.mat')['allSavedTracks'][0]

	# mat = [ele for ele in mat if ele[4][0][0] > 30 and ele[4][0][0] < 50] # only preserve the tracks lasts for more than 60 frames (2 seconds)
	# length of qualified mat: 9
	mat = [ele for ele in mat if ele[4][0][0] > 30]
	mat = mat[:2]

	boundingboxes = [ele[7] for ele in mat]
	frameNum = [ele[6][0] for ele in mat]
	trackNum = len(boundingboxes)
	# boundingboxes = [ele for s in boundingboxes for ele in s]
	# centroids = [ele[:2] for ele in boundingboxes]
	# centroids = [ele[7][:2] for ele in mat]

	angles = []
	for i in range(len(boundingboxes) - 1):
		angle = calculateAngleBetweenTracks(boundingboxes[i], boundingboxes[i + 1])
		angles.append(angle)

	# mosquito starts flying from here; center of the bgimg
	startingPos = np.array([800, 1800])
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
	downScale = 6.0

	resultList = []

	standardWidth = 1920
	standardHeight = 1080

	standardLength = 6500

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

	pastingPosition = [0, 0]
	lastPos = [0, 0]

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

		currTrackPastingPos = []

		if i == 0:
			pastingPosition = list(startingPos)
			currTrackPastingPos.append(pastingPosition)
		else:
			currTrackPastingPos.append(lastPos)

			# temp = currentTrackCentroids[1:]
		for j in range(len(currentTrackCentroids) - 1):
			step = currentTrackCentroids[j + 1] - currentTrackCentroids[j]
			prev = list(currTrackPastingPos[-1])
			# print(step)
			# print(type(step))
			step = step.astype(float)
			step /= downScale
			prev[0] += int(step[1])
			prev[1] += int(step[0])
			currTrackPastingPos.append(prev)

		lastPos = currTrackPastingPos[-1]


		# else:
		# 	previousTrackMovement = processedBoundingBoxes[i - 1][-1] - processedBoundingBoxes[i - 1][0]
		# 	previousTrackMovement = previousTrackMovement[:2]
		# 	previousTrackMovement /= downScale
		# 	pastingPosition[0] += int(previousTrackMovement[1])
		# 	pastingPosition[1] += int(previousTrackMovement[0])


		for frameCounter, frameID in enumerate(currentTrackFrames):

			currFramePastingPos = currTrackPastingPos[frameCounter]

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

			localPatchMask = getLocalPatch(mask, center, radius)
			localPatchImg = getLocalPatch(img, center, radius)


			dst = cv2.resize(localPatchMask, None, fx = 1/downScale, fy = 1/downScale)
			dstImg = cv2.resize(localPatchImg, None, fx = 1/downScale, fy = 1/downScale)


			# co = getObjFromMask(dst)
			# resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)
			co, bbox = getObjFromMaskWhitish(dst)
			if len(co) == 0:
				resultList.append(bgimg)
				labels.append(0)
			else:

				# resulting, positionLabeling = pasteAfterRotationWhitish(dstImg, bgimg, co, pastePosX, pastePosY, bbox, drawRectangleFlag)
				resulting, positionLabeling, errorMsg = pasteWithRescaling(dstImg, bgimg, co, currFramePastingPos[0], currFramePastingPos[1], bbox, drawRectangleFlag)
				print(positionLabeling)
				# resulting, positionLabeling = old(dstImg, bgimg, co, pastePosX, pastePosY, bbox, drawRectangleFlag)
				if len(resulting) > 0:
					resultList.append(resulting)
					labels.append(1)
				else:
					print("shifting invoked!")
					print("error message: ", errorMsg)
					# all cases: 
					# 1: x ok, y < 0
					# 2: x ok, y > 1920
					# 3: x < 0, y ok
					# 4: x > 1080, y ok
					# 5: else (4 corners beyond)
					if errorMsg > 4:
						pass

						# assume: re-start from the middle of the left edge
						# MShift[0][2] += (10 - positionLabeling[0])
						# MShift[1][2] += int(standardHeight / 2 - positionLabeling[1])
					else:
						# assume: for now jump to the opposite side
						if errorMsg == 1:
							# MShift[0][2] += (6000 - 10) # 1910
							# currFrameID = frameCounter
							for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								currTrackPastingPos[currFrameID][1] += 1910
						elif errorMsg == 2:
							# MShift[0][2] -= (6000 - 10)
							for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								currTrackPastingPos[currFrameID][1] -= 1910
						elif errorMsg == 3:
							# MShift[1][2] += (standardHeight - 10) # 1070
							for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								currTrackPastingPos[currFrameID][0] += 1070
						else:
							# MShift[1][2] -= (standardHeight - 10)
							for currFrameID in range(frameCounter, len(currTrackPastingPos)):
								currTrackPastingPos[currFrameID][0] -= 1070



				positionLabels.append(positionLabeling)


			if len(resultList) >= testLength:
				break

		if len(resultList) >= testLength:
			break

	
	np.savetxt('output/testanotheredge.csv', np.array(labels).astype(int), fmt='%i', delimiter=',')

	cap.release()
	cap1.release()


	# fourcc = cv2.FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter('output/testanotheredge.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

	for frame in resultList:
		video.write(frame)


	pass