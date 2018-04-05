#Updated March 2018
#Isha Puri, ishapuri101@gmail.com

#Terminal Command: python forDemo.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import os
import sys
import matplotlib.pyplot as plt
import pandas
from scipy.optimize import curve_fit
import numpy as np

def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
    -P1: a numpy array that consists of the coordinate of the first point (x,y)
    -P2: a numpy array that consists of the coordinate of the second point (x,y)
    -img: the image being processed

    Returns:
    -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


def eyeCenterCoordinates(filePath): #reads eye Center coordinates from Data File
    fp = open(filePath)

    line1 = fp.readline() #first line is the RIGHT EYE
    line2 = fp.readline() #first line is the LEFT EYE

    fp.close()

    arr1 = line1.split(" ")
    arr2 = line2.split(" ")

    rightEyeCenterX = int(arr1[0])
    rightEyeCenterY = int(arr1[1])
    leftEyeCenterX = int(arr2[0])
    leftEyeCenterY = int(arr2[1])

    return (rightEyeCenterX, rightEyeCenterY, leftEyeCenterX, leftEyeCenterY)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


frameCounterProgression = []
rightEyeProgressionX = []
rightEyeProgressionY = []
leftEyeProgressionX = []
leftEyeProgressionY = []

RXMin = sys.maxint
RXMax = -sys.maxint-1
RYMin = sys.maxint
RYMax = -sys.maxint-1
LXMin = sys.maxint
LXMax = -sys.maxint-1
LYMin = sys.maxint
LYMax = -sys.maxint-1


# loop over the frames from the video stream
cap = cv2.VideoCapture('midSaccadeTRIM.mov')
frameCounter = 0
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale

    ret, frame = cap.read()
    if (ret == False):
         break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
    rects = detector(gray, 0)

	# loop over the face detections
    for rect in rects:

        #determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        frameCounter = frameCounter + 1

        # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image

        faceMinX = sys.maxint #these variables find the minimum and maximum values for X and Y, for both left and right eyes
        faceMaxX = -sys.maxint-1		#this is necessary to crop a box around each eye
        faceMinY = sys.maxint
        faceMaxY = -sys.maxint-1

        facialLandmarkCounter = 0

        #KEY:   Right Eye Right Corner = ReRc
        #       Left Eye Right Corner = LeRc    etc.
        ReRcX = 0
        ReRcY = 0
        ReLcX = 0
        ReLcY = 0

        LeRcX = 0
        LeRcY = 0
        LeLcX = 0
        LeLcY = 0

		# loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
        for (x, y) in shape:
            facialLandmarkCounter = facialLandmarkCounter + 1

            if (x < faceMinX):
				faceMinX = x
            if (y < faceMinY):
                faceMinY = y
            if (x > faceMaxX):
                faceMaxX = x
            if (y > faceMaxY):
                faceMaxY = y

            if facialLandmarkCounter == 43: #these numbers identify the right/left corners or right/left eyes
                ReLcX = x
                ReLcY = y
            if facialLandmarkCounter == 46:
                ReRcX = x
                ReRcY = y
            if facialLandmarkCounter == 37:
                LeLcX = x
                LeLcY = y
            if facialLandmarkCounter == 40:
                LeRcX = x
                LeRcY = y

        faceMinY = faceMinY - 180
        faceMaxY = faceMaxY + 50
        faceMinX = faceMinX - 50
        faceMaxX = faceMaxX + 50

        frameHeight, frameWidth, channels = frame.shape
        if ((faceMinY<0) or (faceMaxY>frameHeight) or (faceMinX<0) or (faceMaxX>frameWidth)):
            print("The dimensions are not in the available bounds. Please move back from camera. ")
            break

        justFace = frame[faceMinY:faceMaxY, faceMinX:faceMaxX]
        cv2.imwrite("/Users/rpuri/Desktop/Dyslexia/Final_Programs/faceFrame.jpg", justFace)


        returnValue = os.system("/Users/rpuri/Desktop/eyeLike/build/bin/eyeLike /Users/rpuri/Desktop/Dyslexia/Final_Programs/faceFrame.jpg")
        if (returnValue == -1):
            print("this didn't work")


        (rightEyeCenterX, rightEyeCenterY, leftEyeCenterX, leftEyeCenterY) = eyeCenterCoordinates("/Users/rpuri/Desktop/eyeLike/eyeCenterResults.txt")

        #DEBUG CODE: imageWePassed was used for reference, to see if the points read from the C++ file were correct on the image we passed in.
        #NOTE: Let us show the eyecenters we got w.r.t the frame we passed and ensure it shows correctly
        #imageWePassed = cv2.imread("/Users/rpuri/Desktop/Dyslexia/Final_Programs/faceFrame.jpg",1)
        #cv2.circle(imageWePassed, (rightEyeCenterX, rightEyeCenterY), 2, (0, 255, 0), -1)
        #cv2.circle(imageWePassed, (leftEyeCenterX, leftEyeCenterY), 2, (0, 255, 0), -1)
        #cv2.imshow("frame", frame)
        #cv2.waitKey(0)


        #we are adjusting the eye coordinates with respect to the original bigger video frame (before we cropped it to justFace, which we passed to eyeLike C++ code)
        rightEyeCenterX = rightEyeCenterX + faceMinX
        rightEyeCenterY = rightEyeCenterY + faceMinY
        leftEyeCenterX = leftEyeCenterX + faceMinX
        leftEyeCenterY = leftEyeCenterY + faceMinY

        #draw the centers on the frame for showing later
        #cv2.circle(frame, (rightEyeCenterX, rightEyeCenterY), 0, (0, 255, 0), -1)
        #cv2.circle(frame, (leftEyeCenterX, leftEyeCenterY), 0, (0, 255, 0), -1)


#---------------------------------
        eyeLinePoints = createLineIterator((ReLcX, ReLcY), (ReRcX, ReRcY), gray)

        # Calculate average intensity of pixels along the eyeLinePoints
        averageIntensitySum = 0
        for x in range (0, len(eyeLinePoints)):
            averageIntensitySum += eyeLinePoints[x][2]
        averageIntensity = averageIntensitySum / len(eyeLinePoints)

        # Calculate index in eyeLinePoints for rightEyeCenterX
        ReLcX_index = 0
        ReRcX_index = len(eyeLinePoints) - 1

        currentEyeCenterX_index = 0
        for x in range (0, len(eyeLinePoints)):
            if (eyeLinePoints[x][0] == rightEyeCenterX):
                currentEyeCenterX_index = x
                break

        #cover points with a certain number of pixels before and after the center point with blue
        #to cover any possible error due to glare
        writeBeforeAfter = 8
        for x in range (currentEyeCenterX_index - writeBeforeAfter, currentEyeCenterX_index + writeBeforeAfter):
            #cv2.circle(frame, (eyeLinePoints[x][0], eyeLinePoints[x][1]), 0, (255, 0, 0), -1)
            eyeLinePoints[x][2] = 0


        #set the threshold as the previously found average intensity, hard code if needed
        print('threshold', averageIntensity)
        threshold = averageIntensity * 1.4

        #Start from the current Pupil Center and move to the left until you hit left Iris edge
        currentIndex = currentEyeCenterX_index
        irisLeftEdgeX_index = 0
        while (currentIndex > ReLcX_index) :
            if(eyeLinePoints[currentIndex][2] > threshold):
                irisLeftEdgeX_index = currentIndex + 1
                break
            currentIndex = currentIndex - 1

        #Start from the current Pupil Center and move to the right until you hit right  Iris edge
        currentIndex = currentEyeCenterX_index
        irisRightEdgeX_index = len(eyeLinePoints)-1
        while (currentIndex < ReRcX_index) :
            if(eyeLinePoints[currentIndex][2] > threshold):
                irisRightEdgeX_index = currentIndex - 1
                break
            currentIndex = currentIndex + 1


        if (irisLeftEdgeX_index == 0):
            #Calculate new average for pixels to the left of irisLeftEdgeX_index (which is assumed to be accurate)
            new_average_sum = 0
            for x in range (0, irisRightEdgeX_index):
                new_average_sum += eyeLinePoints[x][2]
            new_average_threshold = (new_average_sum / irisRightEdgeX_index)*1.4
            #Now recalibrate the pixels to the left of irisLeftEdgeX_index for intentsity
            currentIndex = currentEyeCenterX_index
            while (currentIndex > 0) :
                if(eyeLinePoints[currentIndex][2] > new_average_threshold):
                    irisLeftEdgeX_index = currentIndex + 1
                    break
                currentIndex = currentIndex - 1
            #cv2.circle(frame, (eyeLinePoints[irisLeftEdgeX_index][0], eyeLinePoints[irisLeftEdgeX_index][1]), 0, (0, 0, 0), -1) #Black

        midpointXValue = (eyeLinePoints[irisRightEdgeX_index][0] + eyeLinePoints[irisLeftEdgeX_index][0])/2
        midpointYValue = (eyeLinePoints[irisRightEdgeX_index][1] + eyeLinePoints[irisLeftEdgeX_index][1])/2

        print('irisRightEdgeX_index', irisRightEdgeX_index)
        print('irisLeftEdgeX_index', irisLeftEdgeX_index)
        maxAdjustmentPossible = 7
        if (abs(int(midpointXValue)-rightEyeCenterX)<maxAdjustmentPossible):
            rightEyeCenterX = int(midpointXValue)

#-----------------------------------------

        cv2.circle(frame, (ReLcX, ReLcY), 0, (0, 255, 0), -1)
        cv2.circle(frame, (ReRcX, ReRcY), 0, (0, 255, 0), -1)
        cv2.circle(frame, (LeLcX, LeLcY), 0, (0, 255, 0), -1)
        cv2.circle(frame, (LeRcX, LeRcY), 0, (0, 255, 0), -1)

        #It now observes movement from Left corners of respective eyes


        #print("ReRcX, ReRcY", ReRcX, ReRcY)
        #print("ReLcX, ReLcY", ReLcX, ReLcY)
        #print("Right Center X, Y", rightEyeCenterX, rightEyeCenterY)

        #break

        print("Right Eye Center Coordinates: ", rightEyeCenterX, rightEyeCenterY)
        print("Left Eye Center Coordinates: ", leftEyeCenterX, leftEyeCenterY)


        rightEyeCenterX = rightEyeCenterX - ReLcX
        rightEyeCenterY = rightEyeCenterY - ReLcY
        leftEyeCenterX = leftEyeCenterX - LeLcX
        leftEyeCenterY = leftEyeCenterY - LeLcY


        #Creating array for each of the eye centers (left and right eyes, X and Y positions)
        frameCounterProgression.append(frameCounter-1)

        rightEyeProgressionX.append(rightEyeCenterX)
        if (rightEyeCenterX > RXMax):
            RXMax = rightEyeCenterX
        if (rightEyeCenterX < RXMin):
            RXMin = rightEyeCenterX

        rightEyeProgressionY.append(rightEyeCenterY)
        if (rightEyeCenterY > RYMax):
            RYMax = rightEyeCenterY
        if (rightEyeCenterY < RYMin):
            RYMin = rightEyeCenterY

        leftEyeProgressionX.append(leftEyeCenterX)
        if (leftEyeCenterX > LXMax):
            LXMax = leftEyeCenterX
        if (leftEyeCenterX < LXMin):
            LXMin = leftEyeCenterX

        leftEyeProgressionY.append(leftEyeCenterY)
        if (leftEyeCenterY > LYMax):
            LYMax = leftEyeCenterY
        if (leftEyeCenterY < LYMin):
            LYMin = leftEyeCenterY

        cv2.circle(frame, (rightEyeCenterX+ReLcX, (rightEyeProgressionY[0]+ReLcY)), 2, (255, 255, 0), -1)
        print('rightEyeCenterX, (rightEyeProgressionY[0]+ReLcY)', rightEyeCenterX+ReLcX, (rightEyeProgressionY[0]+ReLcY))

    print(frameCounter)
    print("     ")


	# show the frame
    cv2.imshow("frame", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
		break
    #break


newRightEyeProgressionX = rightEyeProgressionX
print('newRightEyeProgressionX', newRightEyeProgressionX)

for x in range (0, len(rightEyeProgressionX)-2):
    if ((rightEyeProgressionX[x+1] > rightEyeProgressionX[x]) and #FILTERS OUT POSITIVE SPIKE
        (rightEyeProgressionX[x+1] > rightEyeProgressionX[x+2])):
        newRightEyeProgressionX[x+1] = rightEyeProgressionX[x]
        newRightEyeProgressionX[x+2] = rightEyeProgressionX[x]

    if ((rightEyeProgressionX[x+1] < rightEyeProgressionX[x]) and #FILTERS OUT NEGATIVE SPIKE
        (rightEyeProgressionX[x+1] < rightEyeProgressionX[x+2])):
        newRightEyeProgressionX[x+1] = rightEyeProgressionX[x]
        newRightEyeProgressionX[x+2] = rightEyeProgressionX[x]

    if ((rightEyeProgressionX[x+1] > rightEyeProgressionX[x]) and   #FILTERS OUT POSITIVE BUMP
        (rightEyeProgressionX[x+2] == rightEyeProgressionX[x+1]) and
        (rightEyeProgressionX[x+3] == rightEyeProgressionX[x])):
        newRightEyeProgressionX[x+1] = rightEyeProgressionX[x]
        newRightEyeProgressionX[x+2] = rightEyeProgressionX[x]

    if ((rightEyeProgressionX[x+1] < rightEyeProgressionX[x]) and   #FILTERS OUT NEGATIVE BUMP
        (rightEyeProgressionX[x+2] == rightEyeProgressionX[x+1]) and
        (rightEyeProgressionX[x+3] == rightEyeProgressionX[x])):
        newRightEyeProgressionX[x+1] = rightEyeProgressionX[x]
        newRightEyeProgressionX[x+2] = rightEyeProgressionX[x]


final_fixation_saccade = []
for x in range(0, len(rightEyeProgressionX)-1):
    if (abs(newRightEyeProgressionX[x] - newRightEyeProgressionX[x+1]))>=2:
        final_fixation_saccade.append(0)
    else:
        final_fixation_saccade.append(1)
final_fixation_saccade.append(0)

print ('final_fixation_saccade', final_fixation_saccade)

difference = -1000000000
fixationDuration = []
for x in range(0, len(final_fixation_saccade)):
    if (final_fixation_saccade[x] == 0):
        for y in range (x+1, len(final_fixation_saccade)):
            if (final_fixation_saccade[y] == 0):
                fixationDuration.append(y-x-1)
                break
print('fixationDuration', fixationDuration)

numFixations = 0
sumFixationLength = 0
for x in range(0,len(fixationDuration)):
    sumFixationLength = sumFixationLength + fixationDuration[x]
    if (fixationDuration[x] != 0):
        numFixations = numFixations + 1
averageFixationLengthFRAMES = (sumFixationLength/numFixations)
averageFixationLengthMS = (sumFixationLength/numFixations)*float(1000/cap.get(cv2.CAP_PROP_FPS))
print('num fixations', numFixations)
print('sumFixationLength', sumFixationLength)
print('average fixation length (ms)', averageFixationLengthMS)
print('averageFixationLengthFRAMES', averageFixationLengthFRAMES)





fps = cap.get(cv2.CAP_PROP_FPS)
print "Frames per second using cap.get(cv2.CAP_PROP_FPS) (where cap is the video captured): {0}".format(fps)

#constantMultiplier = float(1000/fps)
constantMultiplier = 1
print("constantMultiplier", constantMultiplier)

milliSecondProgression = []

for x in frameCounterProgression:
    milliSecondProgression.append(float(float(x)*constantMultiplier))


plt.figure(1)
plt.subplot(221)
plt.plot(milliSecondProgression, newRightEyeProgressionX, 'C0')
plt.axis([0, float(frameCounter)*constantMultiplier, (RXMin), (RXMax)])
plt.title("Right Eye Pupil Progression")
plt.xlabel("Milliseconds")
plt.ylabel("Pixels")

plt.subplot(222)
plt.plot(milliSecondProgression, final_fixation_saccade, 'C0')
plt.axis([0, float(frameCounter)*constantMultiplier, 0, 3])
plt.title("Fixations in Right Eye Pupil Progression")
plt.xlabel("Milliseconds")
plt.ylabel("Fixation")
plt.show()


if ((averageFixationLengthMS>200) and (averageFixationLengthMS<250)):
    print('YOU ARE PROBABLY NOT DYSLEXIC')
if ((averageFixationLengthMS>330) and (averageFixationLengthMS<350)):
    print('YOU ARE PROBABLY DYSLEXIC')
