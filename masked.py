import cv2
# import cv2.cv as cv
import numpy as np
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

if (__name__ == "__main__"):
	bgimg = cv2.imread("images/biology.png")

	h, w, c = bgimg.shape
	newHeight = int(h / 2)
	newWidth = int(w / 2)

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

	resultList = []

	for i in range(frameCount):
		_, img = cap.read()
		_, preMask = cap1.read()
		print("i is: ", i)

		mask = cv2.cvtColor(preMask, cv2.COLOR_BGR2GRAY)

		img = cv2.resize(img, None, fx = 1/6.0, fy = 1/6.0)
		mask = cv2.resize(mask, None, fx = 1/6.0, fy = 1/6.0)

		co = getObjFromMask(mask)

		resulting = paste(img, bgimg, co)

		resultList.append(resulting)

		if i > 3600:
			break

		# res = cv2.bitwise_and(img, img, mask = mask)


	# fourcc = cv2.FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter('fff.mov', fourcc, fps = 30, frameSize = (resWidth, resHeight), isColor = 1)

	for frame in resultList:
		video.write(frame)

	cap.release()
	cap1.release()

	pass