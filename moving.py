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
	return res

def getObjFromMaskWhitish(mask, bodyColorThreshold = 16):
	# print(np.max(mask))
	blurred = []
	body = []
	coordinates = []
	for row in range(len(mask)):
		for col in range(len(mask[0])):
			if mask[row][col] > bodyColorThreshold:
				coordinates.append((row, col))

	return coordinates

def pasteAfterRotationWhitish(capImg, labImg, coordinates, posX, posY):
	res = np.array(labImg).astype(float)
	# res = res.astype(float)
	for (row, col) in coordinates:
		xIndex = int(posX + row)
		yIndex = int(posY + col)
		res[xIndex][yIndex] *= (capImg[row][col] / 255.0)
		# if printFlag:
		# 	print (xIndex, yIndex)
	# print(np.max(res))
	return np.uint8(res)

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

	# source: https://stackoverflow.com/a/31735642/3455398 |  clockwise angle between the two
	def angle_between(p1, p2):
		ang1 = np.arctan2(*p1[::-1])
		ang2 = np.arctan2(*p2[::-1])
		return np.rad2deg((ang1 - ang2) % (2 * np.pi))

	# def unitVector(v):
	# 	return v / np.linalg.norm(v)

	# def angle_between(v1, v2):
	# 	# in radians
	# 	v1U = unitVector(v1)
	# 	v2U = unitVector(v2)

	# 	return np.arccos(np.clip(np.dot(v1U, v2U), -1.0, 1.0))

	return angle_between(firstVector, lastVector)

if (__name__ == "__main__"):
	# bgimg = cv2.imread("images/biology.png")

	mat = scipy.io.loadmat('tracks/MVI_7521_output_multi.mat')['allSavedTracks'][0]

	mat = [ele for ele in mat if ele[4][0][0] > 60] # only preserve the tracks lasts for more than 30 frames (1 second)

	boundingboxes = [ele[7] for ele in mat]
	frameNum = [ele[6][0] for ele in mat]
	trackNum = len(boundingboxes)
	# boundingboxes = [ele for s in boundingboxes for ele in s]
	# centroids = [ele[:2] for ele in boundingboxes]
	# centroids = [ele[7][:2] for ele in mat]

	"""
	Note: first half: 0 - 270: 1; 480 - 960: 1; rest: 0
	      second half: 0 - 200: 1; 930 - end: 1; rest: 0
	"""
	# alternatingFrameIDs = [270, 480, 960]
	containsMosFrameIDRanges = [(0, 270), (480, 960)]
	standardLength = 2500
	containsMosFrameIDRanges = [(0, 200), (580, 720), (930, 1200), (1500, 1650)]

	def containsMosquito(currFrameID):
		for frameRange in containsMosFrameIDRanges:
			if currFrameID >= frameRange[0] and currFrameID <= frameRange[1]:
				return True
		return False


	angles = []
	for i in range(len(boundingboxes) - 1):
		angle = calculateAngleBetweenTracks(boundingboxes[i], boundingboxes[i + 1])
		angles.append(angle)

	# mosquito starts flying from here; center-right of 720p bg video
	startingPos = np.array([150, 150])
	angles = [0] + angles

	# stats:
	# bgimg: 1920 x 1080
	# bgvideo: 720p
	# bgvideo: 30 FPS
	cap = cv2.VideoCapture('videos/MVI_7521.MOV')
	cap1 = cv2.VideoCapture('masks/MVI_7521_output_multi_BG.avi')
	bgcap = cv2.VideoCapture('bgvideos/secondhalf.mov')
	_, bgimg = bgcap.read()
	# fps = cap.get(cv2.CAP_PROP_FPS)
	frameCount = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
	bgFrameCount = int(bgcap.get(cv2.CAP_PROP_FRAME_COUNT))
	bgFrameCounter = 0
	fps = 30

	# resWidth = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
	# resHeight = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
	resHeight, resWidth, c = bgimg.shape
	downScale = 9.0 # 6.0 for 1080p bg videos; 9.0 for 720p bg videos

	resultList = []

	standardWidth = 1920
	standardHeight = 1080

	standardLength = 5500

	pastePosX = startingPos[0]
	pastePosY = startingPos[1]

	testLength = bgFrameCount - 30
	# testLength = 300

	labels = []

	MShift = np.float32([[1, 0, 0], [0, 1, 0]])
	firstCentroidOnCanvas = [0, 0]

	for i in range(trackNum):
		currentTrackFrames = frameNum[i]
		currentTrackCentroids = [ele[:2] for ele in boundingboxes[i]]
		assert len(currentTrackFrames) == len(currentTrackCentroids), "number of frames not matching with the number of centroids"
		firstCentroid = currentTrackCentroids[0]
		lastCentroid = currentTrackCentroids[-1]
		movementVector = np.array(lastCentroid) - np.array(firstCentroid)

		"""
		get the shift distance between the two tracks
		"""
		# MShift = np.float32([[1, 0, 0], [0, 1, 0]])
		# pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
		# pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)
		if i > 0:
			previousEndingCentroid = boundingboxes[i - 1][-1][:2]
			xShift = firstCentroid[0] - previousEndingCentroid[0]
			yShift = firstCentroid[1] - previousEndingCentroid[1]
			# MShift = np.float32([[1, 0, -xShift], [0, 1, -yShift]])

			previousStartingCentroid = boundingboxes[i - 1][0][:2]
			previousTrackShiftX = previousEndingCentroid[0] - previousStartingCentroid[0]
			previousTrackShiftY = previousEndingCentroid[1] - previousStartingCentroid[1]
			# MShift = np.float32([[1, 0, -xShift], [0, 1, -yShift]])
			MShift[0][2] -= xShift
			MShift[1][2] -= yShift

			firstCentroidOnCanvas[0] += previousTrackShiftX
			firstCentroidOnCanvas[1] += previousTrackShiftY
			
			# pastePosX -= int(xShift / downScale)
			# pastePosY -= int(yShift / downScale)
			# nextStartingCentroid = boundingboxes[i + 1][0]

		else:
			# first track: mosquito starts flying from center of the image
			pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
			pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)

			# transformedFirstCentroid = firstCentroid
			# mappedCentroidY = int(standardLength /2 - standardWidth/2) + transformedFirstCentroid[0]
			# mappedCentroidX = int(standardLength /2 - standardHeight/2) + transformedFirstCentroid[1]
			firstCentroidOnCanvas[0] = int(standardLength /2 - standardWidth/2) + firstCentroid[0]
			firstCentroidOnCanvas[1] = int(standardLength /2 - standardHeight/2) + firstCentroid[1]

		# frameCounter = 0

		rotationDegree = angles[i]
		M = cv2.getRotationMatrix2D((firstCentroid[0], firstCentroid[1]), rotationDegree, 1)
		for frameCounter, frameID in enumerate(currentTrackFrames):
			cap1.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, preMask = cap1.read()
			mask = cv2.cvtColor(preMask, cv2.COLOR_BGR2GRAY)
			cap.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, img = cap.read()
			radius = int(1.0 * max(boundingboxes[i][frameCounter][-1], boundingboxes[i][frameCounter][-2]))
			center = currentTrackCentroids[frameCounter]


			mask = clearUnrelatedMasks(mask, center, radius)


			'''
			Start of logic 1: rotate the whole mask and img by the angle, and paste it to bgimg
			'''
			largerImg = np.zeros(shape = (standardLength, standardLength, 3))
			largerMask = np.zeros(shape = (standardLength, standardLength))

			largerImg[int(len(largerImg) /2 - standardHeight/2) : int(len(largerImg) / 2 + standardHeight/2), int(len(largerImg) /2 - standardWidth/2) : int(len(largerImg) / 2 + standardWidth/2)] = img
			largerMask[int(len(largerImg) /2 - standardHeight/2) : int(len(largerImg) / 2 + standardHeight/2), int(len(largerImg) /2 - standardWidth/2) : int(len(largerImg) / 2 + standardWidth/2)] = mask

			img = largerImg
			mask = largerMask

			try:
				dst = cv2.warpAffine(mask, MShift, (standardLength, standardLength))
				dstImg = cv2.warpAffine(img, MShift, (standardLength, standardLength))
			except:
				print("mosquitoes moving to the edge in raw video, jumping to the next track")
				break

			dst = cv2.resize(dst, None, fx = 1/downScale, fy = 1/downScale)
			dstImg = cv2.resize(dstImg, None, fx = 1/downScale, fy = 1/downScale)

			_, bgimg = bgcap.read()
			if containsMosquito(bgFrameCounter):
				

				bgFrameCounter += 1
				co = getObjFromMaskWhitish(dst)
				resulting = pasteAfterRotationWhitish(dstImg, bgimg, co, pastePosX, pastePosY)
				resultList.append(resulting)
				# resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)
				labels.append(1)
				# resultList.append(resulting)

			else:
				while not containsMosquito(bgFrameCounter):
					_, temp = bgcap.read()
					resultList.append(temp)
					bgFrameCounter += 1
					labels.append(0)

					if len(resultList) >= testLength:
						break

			print(len(resultList))


			if len(resultList) >= testLength:
				break

		if len(resultList) >= testLength:
			break




	"""
	Start of adding another mosquitoes
	"""
	boundingboxes = boundingboxes[::-1]
	frameNum = frameNum[::-1]

	startingPos = [300, 450]
	bgFrameCounter = 0

	MShift = np.float32([[1, 0, 0], [0, 1, 0]])
	firstCentroidOnCanvas = [0, 0]

	for i in range(trackNum):
		currentTrackFrames = frameNum[i]
		currentTrackCentroids = [ele[:2] for ele in boundingboxes[i]]
		assert len(currentTrackFrames) == len(currentTrackCentroids), "number of frames not matching with the number of centroids"
		firstCentroid = currentTrackCentroids[0]
		lastCentroid = currentTrackCentroids[-1]
		movementVector = np.array(lastCentroid) - np.array(firstCentroid)

		"""
		get the shift distance between the two tracks
		"""
		# MShift = np.float32([[1, 0, 0], [0, 1, 0]])
		# pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
		# pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)
		if i > 0:
			previousEndingCentroid = boundingboxes[i - 1][-1][:2]
			xShift = firstCentroid[0] - previousEndingCentroid[0]
			yShift = firstCentroid[1] - previousEndingCentroid[1]
			# MShift = np.float32([[1, 0, -xShift], [0, 1, -yShift]])

			previousStartingCentroid = boundingboxes[i - 1][0][:2]
			previousTrackShiftX = previousEndingCentroid[0] - previousStartingCentroid[0]
			previousTrackShiftY = previousEndingCentroid[1] - previousStartingCentroid[1]
			# MShift = np.float32([[1, 0, -xShift], [0, 1, -yShift]])
			MShift[0][2] -= xShift
			MShift[1][2] -= yShift

			firstCentroidOnCanvas[0] += previousTrackShiftX
			firstCentroidOnCanvas[1] += previousTrackShiftY
			
			# pastePosX -= int(xShift / downScale)
			# pastePosY -= int(yShift / downScale)
			# nextStartingCentroid = boundingboxes[i + 1][0]

		else:
			# first track: mosquito starts flying from center of the image
			pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
			pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)

			# transformedFirstCentroid = firstCentroid
			# mappedCentroidY = int(standardLength /2 - standardWidth/2) + transformedFirstCentroid[0]
			# mappedCentroidX = int(standardLength /2 - standardHeight/2) + transformedFirstCentroid[1]
			firstCentroidOnCanvas[0] = int(standardLength /2 - standardWidth/2) + firstCentroid[0]
			firstCentroidOnCanvas[1] = int(standardLength /2 - standardHeight/2) + firstCentroid[1]



		rotationDegree = angles[i]
		M = cv2.getRotationMatrix2D((firstCentroid[0], firstCentroid[1]), rotationDegree, 1)
		for frameCounter, frameID in enumerate(currentTrackFrames):
			cap1.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, preMask = cap1.read()
			mask = cv2.cvtColor(preMask, cv2.COLOR_BGR2GRAY)
			cap.set(cv2.CAP_PROP_POS_FRAMES, frameID - 1)
			_, img = cap.read()
			radius = int(1.0 * max(boundingboxes[i][frameCounter][-1], boundingboxes[i][frameCounter][-2]))
			center = currentTrackCentroids[frameCounter]


			mask = clearUnrelatedMasks(mask, center, radius)

			'''
			Start of logic 1: rotate the whole mask and img by the angle, and paste it to bgimg
			'''
			largerImg = np.zeros(shape = (standardLength, standardLength, 3))
			largerMask = np.zeros(shape = (standardLength, standardLength))

			largerImg[int(len(largerImg) /2 - standardHeight/2) : int(len(largerImg) / 2 + standardHeight/2), int(len(largerImg) /2 - standardWidth/2) : int(len(largerImg) / 2 + standardWidth/2)] = img
			largerMask[int(len(largerImg) /2 - standardHeight/2) : int(len(largerImg) / 2 + standardHeight/2), int(len(largerImg) /2 - standardWidth/2) : int(len(largerImg) / 2 + standardWidth/2)] = mask

			img = largerImg
			mask = largerMask

			try:
				dst = cv2.warpAffine(mask, MShift, (standardLength, standardLength))
				dstImg = cv2.warpAffine(img, MShift, (standardLength, standardLength))
			except:
				print("mosquitoes moving to the edge in raw video, jumping to the next track")
				break

			dst = cv2.resize(dst, None, fx = 1/downScale, fy = 1/downScale)
			dstImg = cv2.resize(dstImg, None, fx = 1/downScale, fy = 1/downScale)

			# co = getObjFromMask(dst)

			# _, bgimg = bgcap.read()
			bgimg = resultList[bgFrameCounter]

			if containsMosquito(bgFrameCounter):
				# _, bgimg = bgcap.read()

				
				co = getObjFromMaskWhitish(dst)
				resulting = pasteAfterRotationWhitish(dstImg, bgimg, co, pastePosX, pastePosY)
				resultList[bgFrameCounter] = resulting
				bgFrameCounter += 1
				# resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)
				# labels.append(1)

			# if containsMosquito(bgFrameCounter):

				
			# 	resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)
			# 	# labels.append(1)
			# 	# resultList.append(resulting)
			# 	resultList[bgFrameCounter] = resulting
			# 	bgFrameCounter += 1

			else:
				while not containsMosquito(bgFrameCounter):
					# _, temp = bgcap.read()
					# resultList.append(temp)
					bgFrameCounter += 1
					# labels.append(0)

					if bgFrameCounter >= testLength:
						break


			# print(len(resultList))
			print(bgFrameCounter)


			if bgFrameCounter >= testLength:
				break

		if bgFrameCounter >= testLength:
			break



	"""
	End of adding another mosquitoes, so you can remove the whole section to add only one mosquito
	"""


	np.savetxt('output/labels-two.csv', np.array(labels).astype(int), fmt='%i', delimiter=',')


	cap.release()
	cap1.release()
	bgcap.release()


	# fourcc = cv2.FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter('output/moving-two.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

	for frame in resultList:
		video.write(frame)


	pass