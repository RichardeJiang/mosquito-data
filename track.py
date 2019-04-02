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
	bgimg = cv2.imread("images/biology.png")

	mat = scipy.io.loadmat('tracks/MVI_7521_output_multi.mat')['allSavedTracks'][0]

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
	startingPos = np.array([540, 960])
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

	pastePosX = startingPos[0]
	pastePosY = startingPos[1]

	testLength = 900

	for i in range(trackNum):
		currentTrackFrames = frameNum[i]
		currentTrackCentroids = [ele[:2] for ele in boundingboxes[i]]
		assert len(currentTrackFrames) == len(currentTrackCentroids), "number of frames not matching with the number of centroids"
		firstCentroid = currentTrackCentroids[0]
		lastCentroid = currentTrackCentroids[-1]
		movementVector = np.array(lastCentroid) - np.array(firstCentroid)


		# pastePosX = startingPos[0] - int(firstCentroid[1] / downScale)
		# pastePosY = startingPos[1] - int(firstCentroid[0] / downScale)

		# frameCounter = 0

		rotationDegree = angles[i]
		# M = cv2.getRotationMatrix2D((firstCentroid[0], firstCentroid[1]), rotationDegree, 1)
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
			Start of logic 2: segment the mosquito out completely, calculate the position of the center on bgimg, and paste
			'''

			zoomInMask = mask[center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius]
			zoomInImg = img[center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius]

			M = cv2.getRotationMatrix2D((radius, radius), rotationDegree, 1)

			try:

				rotatedMask = cv2.warpAffine(zoomInMask, M, (radius * 2, radius * 2))
				rotatedImg = cv2.warpAffine(zoomInImg, M, (radius * 2, radius * 2))

			except:
				print("mosquitoes moving to the edge, jumping to the next track")
				break

			rotatedMask = cv2.resize(rotatedMask, None, fx = 1/downScale, fy = 1/downScale)
			rotatedImg = cv2.resize(rotatedImg, None, fx = 1/downScale, fy = 1/downScale)

			co = getObjFromMask(rotatedMask)
			resulting = pasteWithRadius(rotatedImg, bgimg, co, pastePosX, pastePosY, radius)


			resultList.append(resulting)


			print(len(resultList))


			# calculate the position of the next centroid on the bgimg
			if frameCounter < len(currentTrackFrames) - 1:
				nextCenter = currentTrackCentroids[frameCounter + 1]
				movementVector = np.array(nextCenter) - np.array(center)
				movementVector = movementVector / downScale
				rotationMatrix = M[:, :2].T
				finalMovement = np.dot(rotationMatrix, movementVector)
				pastePosX += int(finalMovement[1])
				pastePosY += int(finalMovement[0])
			# resulting = pasteAfterRotation(capImg, labImg, coordinates, posX, posY)



			if len(resultList) >= testLength:
				break

		if len(resultList) >= testLength:
			break

	labels = [1] * len(resultList)

	negatives = [bgimg] * testLength
	resultList += negatives
	negativesLabels = [0] * testLength
	labels += negativesLabels
	np.savetxt('labels.csv', np.array(labels), delimiter=',')




	cap.release()
	cap1.release()



	# for i in range(frameCount):
	# 	_, img = cap.read()
	# 	_, preMask = cap1.read()
	# 	print("i is: ", i)

	# 	mask = cv2.cvtColor(preMask, cv2.COLOR_BGR2GRAY)

	# 	img = cv2.resize(img, None, fx = 1/6.0, fy = 1/6.0)
	# 	mask = cv2.resize(mask, None, fx = 1/6.0, fy = 1/6.0)

	# 	co = getObjFromMask(mask)

	# 	resulting = paste(img, bgimg, co)

	# 	resultList.append(resulting)

	# 	if i > 3600:
	# 		break



	# fourcc = cv2.FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter('sharper.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

	for frame in resultList:
		video.write(frame)

	

	pass