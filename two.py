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

	# source: https://stackoverflow.com/a/31735642/3455398 |  clockwise angle between the two
	def angle_between(p2, p1):
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
	bgimg = cv2.imread("images/biology.png")

	mat = scipy.io.loadmat('tracks/MVI_7521_output_multi.mat')['allSavedTracks'][0]

	mat = [ele for ele in mat if ele[4][0][0] > 60] # only preserve the tracks lasts for more than 60 frames (2 seconds)
	# mat = mat[:4] # for testing purposes
	# mat = mat[3:]

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
	startingPos = np.array([200, 200])
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

	standardLength = 5000

	pastePosX = startingPos[0]
	pastePosY = startingPos[1]

	testLength = 1200

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
		
		# pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
		# pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)
		if i > 0:
			previousEndingCentroid = boundingboxes[i - 1][-1][:2]
			xShift = firstCentroid[0] - previousEndingCentroid[0]
			yShift = firstCentroid[1] - previousEndingCentroid[1]

			previousStartingCentroid = boundingboxes[i - 1][0][:2]
			previousTrackShiftX = previousEndingCentroid[0] - previousStartingCentroid[0]
			previousTrackShiftY = previousEndingCentroid[1] - previousStartingCentroid[1]
			# MShift = np.float32([[1, 0, -xShift], [0, 1, -yShift]])
			MShift[0][2] -= xShift
			MShift[1][2] -= yShift

			firstCentroidOnCanvas[0] += previousTrackShiftX
			firstCentroidOnCanvas[1] += previousTrackShiftY
			
			# transformedFirstCentroid = np.dot(MShift, np.array([firstCentroid[0], firstCentroid[1], 1]))
			# transformedFirstCentroid = previousEndingCentroid
			# pastePosX -= int(yShift / downScale)
			# pastePosY -= int(xShift / downScale)
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
		# mappedCentroidY = int(standardLength /2 - standardWidth/2) + transformedFirstCentroid[0]
		# mappedCentroidX = int(standardLength /2 - standardHeight/2) + transformedFirstCentroid[1]
		# M = cv2.getRotationMatrix2D((mappedCentroidY, mappedCentroidX), rotationDegree, 1)
		for frameCounter, frameID in enumerate(currentTrackFrames):
			# if frameCounter < 1 or frameCounter > len(currentTrackFrames) - 2:
			# 	printFlag = True
			# 	if frameCounter < 1:
			# 		print("start")
			# 	else:
			# 		print("end")
			# 	print "start" if frameCounter < 1 else "end"
			# else:
			# 	printFlag = False
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

				# dst = cv2.warpAffine(dst, M, (standardWidth, standardHeight))
				# dstImg = cv2.warpAffine(dstImg, M, (standardWidth, standardHeight))

				
			except:
				print("mosquitoes moving to the edge in raw video, jumping to the next track")
				break

			dst = cv2.resize(dst, None, fx = 1/downScale, fy = 1/downScale)
			dstImg = cv2.resize(dstImg, None, fx = 1/downScale, fy = 1/downScale)

			# co = getObjFromMask(dst)
			# resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)
			co = getObjFromMaskWhitish(dst)
			resulting = pasteAfterRotationWhitish(dstImg, bgimg, co, pastePosX, pastePosY)

			resultList.append(resulting)


			print(len(resultList))


			if len(resultList) >= testLength:
				break

		if len(resultList) >= testLength:
			break


	boundingboxes = boundingboxes[::-1]
	frameNum = frameNum[::-1]
	angles = angles[::-1]
	angles = [-ele for ele in angles]

	startingPos = [600, 1000]
	totalCounter = 0
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
		
		# pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
		# pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)
		if i > 0:
			previousEndingCentroid = boundingboxes[i - 1][-1][:2]
			xShift = firstCentroid[0] - previousEndingCentroid[0]
			yShift = firstCentroid[1] - previousEndingCentroid[1]

			previousStartingCentroid = boundingboxes[i - 1][0][:2]
			previousTrackShiftX = previousEndingCentroid[0] - previousStartingCentroid[0]
			previousTrackShiftY = previousEndingCentroid[1] - previousStartingCentroid[1]
			# MShift = np.float32([[1, 0, -xShift], [0, 1, -yShift]])
			MShift[0][2] -= xShift
			MShift[1][2] -= yShift

			firstCentroidOnCanvas[0] += previousTrackShiftX
			firstCentroidOnCanvas[1] += previousTrackShiftY
			
			# transformedFirstCentroid = np.dot(MShift, np.array([firstCentroid[0], firstCentroid[1], 1]))
			# transformedFirstCentroid = previousEndingCentroid
			# pastePosX -= int(yShift / downScale)
			# pastePosY -= int(xShift / downScale)
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
		# mappedCentroidY = int(standardLength /2 - standardWidth/2) + transformedFirstCentroid[0]
		# mappedCentroidX = int(standardLength /2 - standardHeight/2) + transformedFirstCentroid[1]
		# M = cv2.getRotationMatrix2D((mappedCentroidY, mappedCentroidX), rotationDegree, 1)
		# M = cv2.getRotationMatrix2D((firstCentroid[0], firstCentroid[1]), rotationDegree, 1)
		for frameCounter, frameID in enumerate(currentTrackFrames):
			# if frameCounter < 1 or frameCounter > len(currentTrackFrames) - 2:
			# 	printFlag = True
			# 	if frameCounter < 1:
			# 		print("start")
			# 	else:
			# 		print("end")
			# 	print "start" if frameCounter < 1 else "end"
			# else:
			# 	printFlag = False
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

				# dst = cv2.warpAffine(dst, M, (standardWidth, standardHeight))
				# dstImg = cv2.warpAffine(dstImg, M, (standardWidth, standardHeight))

				
			except:
				print("mosquitoes moving to the edge in raw video, jumping to the next track")
				break

			dst = cv2.resize(dst, None, fx = 1/downScale, fy = 1/downScale)
			dstImg = cv2.resize(dstImg, None, fx = 1/downScale, fy = 1/downScale)

			# co = getObjFromMask(dst)
			# bgimg = resultList[totalCounter]
			# resulting = pasteAfterRotation(dstImg, bgimg, co, pastePosX, pastePosY)

			co = getObjFromMaskWhitish(dst)
			bgimg = resultList[totalCounter]
			resulting = pasteAfterRotationWhitish(dstImg, bgimg, co, pastePosX, pastePosY)

			resultList[totalCounter] = resulting
			totalCounter += 1

			print(totalCounter)


			if totalCounter >= testLength:
				break

		if totalCounter >= testLength:
			break

	# labels = [1] * len(resultList)

	# negatives = [bgimg] * testLength
	# resultList += negatives
	# negativesLabels = [0] * testLength
	# labels += negativesLabels
	# np.savetxt('labels-smoother.csv', np.array(labels), delimiter=',')


	cap.release()
	cap1.release()


	print("writing videos...")
	# fourcc = cv2.FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter('output/stitchingtwo.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

	for frame in resultList:
		video.write(frame)


	pass