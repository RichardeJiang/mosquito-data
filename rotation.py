import cv2
# import cv2.cv as cv
import numpy as np
import scipy.io
import os

def extractObj(capImg, bgImg, newWidth, newHeight):
	res = np.zeros(shape = capImg.shape)
	check = np.zeros(shape=(res.shape[0], res.shape[1]))
	can = np.absolute(capImg - bgImg)
	for row in range(len(can)):
		for col in range(len(can[0])):
			if sum(can[row][col]) >= 250:
				res[row][col] = capImg[row][col]
				check[row][col] = 1
	
	mainBoard = np.zeros(shape=(newHeight, newWidth))
	mainBoard[newHeight / 2 - res.shape[0] : newHeight / 2 + res.shape[0], newWidth / 2 - res.shape[1] : newWidth / 2 + res.shape[1]] = check
	return res, mainBoard

def getObj(capImg, bgImg):
	# res = np.zeros(shape = labImg.shape)
	# res = np.array(labImg)
	coordinates = []
	can = np.absolute(capImg - bgImg)
	for row in range(len(can)):
		for col in range(len(can[0])):
			if sum(can[row][col]) >= 720:
				# xIndex = int(labImg.shape[0] / 2 - res.shape[0] / 2 + row)
				# yIndex = int(labImg.shape[1] / 2 - res.shape[1] / 2 + col)
				# res[xIndex][yIndex] = capImg[row][col]
				coordinates.append((row, col))
	return coordinates

def getObjFromMask(mask):
	coordinates = []
	for row in range(len(mask)):
		for col in range(len(mask[0])):
			if mask[row][col] > 0:
				coordinates.append((row, col))
	return coordinates

"""
TODO:
current pasting logic: for each pixel, multiplying the mosquito pixel with the background pixel
maybe use the more precise logic: for the center body: replace; while for the whitish surroundings, use multiplication
Note: here the bodyColorThreshold = 16 is predefined by the previous tryer
Note 2: Nitin's old logic is wrong, where im2uint8 is used: it'll scale the whole image
"""

def getObjFromMaskWhitish(mask, bodyColorThreshold = 8):
	# print(np.max(mask))
	blurred = []
	body = []
	coordinates = []
	# checkingThreshold = mask > bodyColorThreshold
	checkingThreshold = np.argwhere(mask > bodyColorThreshold)
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

def getObjBoth(mask, bodyColorThreshold = 16):
	# print(np.max(mask))
	blurred = []
	body = []
	coordinates = []
	# checkingThreshold = mask > bodyColorThreshold
	checkingThreshold = np.argwhere(mask >= bodyColorThreshold)
	body = np.argwhere(mask >= bodyColorThreshold)
	blur = np.argwhere(mask < bodyColorThreshold and mask > 0)
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
	# return res.astype(int)


def paste(capImg, labImg, coordinates):
	res = np.array(labImg)
	for (row, col) in coordinates:
		xIndex = int(labImg.shape[0] / 2 - capImg.shape[0] / 2 + row)
		yIndex = int(labImg.shape[1] / 2 - capImg.shape[1] / 2 + col)
		res[xIndex][yIndex] = capImg[row][col]
	return res

def pasteAfterRotation(capImg, labImg, coordinates, posX, posY):
	res = np.array(labImg)
	for (row, col) in coordinates:
		xIndex = int(posX + row)
		yIndex = int(posY + col)
		res[xIndex][yIndex] = capImg[row][col]
		# if printFlag:
		# 	print (xIndex, yIndex)
	return res

def pasteWithRadius(capImg, labImg, coordinates, posX, posY, radius):
	res = np.array(labImg)
	for (row, col) in coordinates:
		xIndex = int(posX - radius + row)
		yIndex = int(posY - radius + col)
		res[xIndex][yIndex] = capImg[row][col]
	return res

def clearUnrelatedMasks(mask, centroid, radius):
	res = np.zeros(shape = mask.shape)
	radius = int(radius)
	# print(mask)
	# print(centroid)
	res[centroid[1] - radius : centroid[1] + radius, centroid[0] - radius : centroid[0] + radius] = \
	mask[centroid[1] - radius : centroid[1] + radius, centroid[0] - radius : centroid[0] + radius]

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

if (__name__ == "__main__"):
	bgimg = cv2.imread("images/biology.png")

	mat = scipy.io.loadmat('tracks/MVI_7521_output_multi.mat')['allSavedTracks'][0]

	mat = [ele for ele in mat if ele[4][0][0] > 30 and ele[4][0][0] < 50] # only preserve the tracks lasts for more than 60 frames (2 seconds)
	# length of qualified mat: 9
	# mat = mat[2:4] # for testing purposes
	# mat = mat[2:5]
	mat = mat[3:7]

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

	# lastone = calculateAngleBetweenTracks(boundingboxes[len(boundingboxes) - 1], boundingboxes[0])
	# angles.append(lastone)
	# mosquito starts flying from here; center of the bgimg
	startingPos = np.array([200, 400])
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

	for i in range(trackNum):
		currentTrackFrames = frameNum[i]
		currentTrackCentroids = [ele[:2] for ele in boundingboxes[i]]
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

		def degToRad(ang):
			return ang * np.pi / 180


		if i > 0:

			firstCentroidOnCanvas = list(lastCentroidOnCanvas)
			previousEndingCentroid = boundingboxes[i - 1][-1][:2]
			previousStartingCentroid = boundingboxes[i - 1][0][:2]

			"""
			The following part calculates the average moving distance of the last vector of previous track,
			and the first vector of the current track
			so shift the MShift by this distance to ensure a smooth transition between tracks
			"""
			previousLastVector = [0, 0]
			previousLastVector[0] = boundingboxes[i - 1][-1][0] - boundingboxes[i - 1][-2][0]
			previousLastVector[1] = boundingboxes[i - 1][-1][1] - boundingboxes[i - 1][-2][1]
			previousStepLength = np.sqrt(previousLastVector[0]**2+previousLastVector[1]**2)
			currentFirstVector = [0, 0]
			currentFirstVector[0] = boundingboxes[i][2][0] - boundingboxes[i][1][0]
			currentFirstVector[1] = boundingboxes[i][2][1] - boundingboxes[i][1][1]
			currentStepLength = np.sqrt(currentFirstVector[0]**2+currentFirstVector[1]**2)
			movementRatio = (currentStepLength + previousStepLength) / 2 / previousStepLength

			lastStepVectorOnCanvas = np.dot(tempRotationMatrix, np.array(previousLastVector))
			newStepVectorOnCanvas = lastStepVectorOnCanvas * movementRatio
			firstCentroidOnCanvas[0] += newStepVectorOnCanvas[0]
			firstCentroidOnCanvas[1] += newStepVectorOnCanvas[1]


			"""
			end of smoothing transition code
			"""

			previousMovementVector = np.array(previousEndingCentroid) - np.array(previousStartingCentroid)
			currentRotationMatrix = np.array([[np.cos(degToRad(rotationDegree)), -np.sin(degToRad(rotationDegree))], [np.sin(degToRad(rotationDegree)), np.cos(degToRad(rotationDegree))]])
			tempRotationMatrix = np.dot(tempRotationMatrix, currentRotationMatrix)
			print("current angle matrix: ", currentRotationMatrix)
			print("cumulated angle matrix: ", tempRotationMatrix)
			currentMovementVectorOnCanvas = np.dot(tempRotationMatrix, movementVector)
			print("calculated movement on canvas: ", currentMovementVectorOnCanvas)
			# print()
			lastCentroidOnCanvas[0] += int(currentMovementVectorOnCanvas[0])
			lastCentroidOnCanvas[1] += int(currentMovementVectorOnCanvas[1])

			theta += rotationDegree
			print("cumulated angle: ", theta)

			# firstStepVector = np.dot(tempRotationMatrix, currentFirstVector)


		else:
			firstCentroidOnCanvas[0] = int(standardLength /2 - standardWidth/2) + firstCentroid[0]
			firstCentroidOnCanvas[1] = int(standardLength /2 - standardHeight/2) + firstCentroid[1]
			lastCentroidOnCanvas[0] = firstCentroidOnCanvas[0] + movementVector[0]
			lastCentroidOnCanvas[1] = firstCentroidOnCanvas[1] + movementVector[1]


		initialFirstCentroidOnCanvas = np.array([int(standardLength /2 - standardWidth/2) + firstCentroid[0], int(standardLength /2 - standardHeight/2) + firstCentroid[1]])
		xShift = firstCentroidOnCanvas[0] - initialFirstCentroidOnCanvas[0]
		yShift = firstCentroidOnCanvas[1] - initialFirstCentroidOnCanvas[1]
		MShift = np.float32([[1, 0, xShift], [0, 1, yShift]])
		print("initial centroid on canvas: ", initialFirstCentroidOnCanvas)
		print("first centroid on canvas: ", firstCentroidOnCanvas)
		print("last centroid on canvas: ", lastCentroidOnCanvas)
		print("shifting matrix: ", np.array([[1, 0, int(xShift)], [0, 1, int(yShift)]]))
		M = cv2.getRotationMatrix2D((firstCentroidOnCanvas[0], firstCentroidOnCanvas[1]), -theta, 1)






		"""
		get the shift distance between the two tracks
		"""

		# if i > 0:
		# 	previousEndingCentroid = boundingboxes[i - 1][-1][:2]
		# 	xShift = firstCentroid[0] - previousEndingCentroid[0]
		# 	yShift = firstCentroid[1] - previousEndingCentroid[1]

		# 	previousStartingCentroid = boundingboxes[i - 1][0][:2]
		# 	previousTrackShiftX = previousEndingCentroid[0] - previousStartingCentroid[0]
		# 	previousTrackShiftY = previousEndingCentroid[1] - previousStartingCentroid[1]
		# 	MShift[0][2] -= xShift
		# 	MShift[1][2] -= yShift

		# 	firstCentroidOnCanvas[0] += previousTrackShiftX
		# 	firstCentroidOnCanvas[1] += previousTrackShiftY


		# else:
		# 	# first track: mosquito starts flying from center of the image
		# 	pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
		# 	pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)

		# 	firstCentroidOnCanvas[0] = int(standardLength /2 - standardWidth/2) + firstCentroid[0]
		# 	firstCentroidOnCanvas[1] = int(standardLength /2 - standardHeight/2) + firstCentroid[1]


		# rotationDegree = angles[i]

		# M = cv2.getRotationMatrix2D((firstCentroidOnCanvas[0], firstCentroidOnCanvas[1]), rotationDegree, 1)

		for frameCounter, frameID in enumerate(currentTrackFrames):

			cap1.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, preMask = cap1.read()
			mask = cv2.cvtColor(preMask, cv2.COLOR_BGR2GRAY)
			cap.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, img = cap.read()
			bkimg = np.array(img)
			radius = int(1.0 * max(boundingboxes[i][frameCounter][-1], boundingboxes[i][frameCounter][-2]))
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

				dst = cv2.warpAffine(dst, M, (standardLength, standardLength))
				dstImg = cv2.warpAffine(dstImg, M, (standardLength, standardLength))

				
			except:
				print("mosquitoes moving to the edge in raw video, jumping to the next track")
				break

			

			dst = cv2.resize(dst, None, fx = 1/downScale, fy = 1/downScale)
			dstImg = cv2.resize(dstImg, None, fx = 1/downScale, fy = 1/downScale)


			# co = getObjFromMask(dst)
			# resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)
			co, bbox = getObjFromMaskWhitish(dst)
			resulting, positionLabeling = pasteAfterRotationWhitish(dstImg, bgimg, co, pastePosX, pastePosY, bbox, drawRectangleFlag)
			resultList.append(resulting)

			positionLabels.append(positionLabeling)

			if len(resultList) < 30:
				cv2.imwrite('output/images/mask-' + str(len(resultList)) + '.png', dst)
				cv2.imwrite('output/images/raw-' + str(len(resultList)) + '.png', dstImg)
				cv2.imwrite('output/images/composite-' + str(len(resultList)) + '.png', resulting)
				cv2.imwrite('output/images/clear-' + str(len(resultList)) + '.png', bkimg)


			# print(len(resultList))


			if len(resultList) >= testLength:
				break

		if len(resultList) >= testLength:
			break

	# labels = [1] * len(resultList)

	# negatives = [bgimg] * testLength
	# resultList += negatives
	# negativesLabels = [0] * testLength
	# labels += negativesLabels
	# np.savetxt('labels-smoother.csv', np.array(labels), delimiter=',')
	np.savetxt('output/positions.csv', np.array(positionLabels).astype(int), fmt='%i', delimiter=',')

	cap.release()
	cap1.release()


	# fourcc = cv2.FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter('output/print.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

	for frame in resultList:
		video.write(frame)


	pass