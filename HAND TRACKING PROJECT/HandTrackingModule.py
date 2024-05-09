import cv2
import mediapipe as mp
import time

# These are the parameters which we are passed so we can customIze that
# ''' 
#     static_image_mode = False,
#     max_num_hands = 2,
#     min_detection_confidence = 0.5,
#     min_tracking_confidence = 0.5
    
# '''

class HandDetector():

    def __init__(self, mode = False, MaxHands = 2, DetectionConfidence = 0.5, TrackConfidence = 0.5):
        self.mode = mode # Create a object and the object have its own vaiable
        self.MaxHands = MaxHands
        self.DetectionConfidence = DetectionConfidence
        self.TrackConfidence = TrackConfidence


        # Functionality for Detecting and tracking hands in images or video
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.MaxHands, self.DetectionConfidence, self.TrackConfidence)

        # Draw points in hands
        self.mpDraw = mp.solutions.drawing_utils 


    def FindHands(self, img, Draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # to check if there are multiple hands
        #print(results.multi_hand_landmarks)

        # For checking points in every hand
        if self.results.multi_hand_landmarks:
            for handlandmarks in self.results.multi_hand_landmarks:
                if Draw:
                    self.mpDraw.draw_landmarks(img, handlandmarks, self.mpHands.HAND_CONNECTIONS)

        return img
    

    def FindPosition(self, img, HandNo = 0, Draw = True):

        # This list return all the landmnark positions
        LandmarkList = []

        if self.results.multi_hand_landmarks:
            MyHand = self.results.multi_hand_landmarks[HandNo]

            # For id numer of each points in hands
            for id, lm in enumerate(MyHand.landmark):

                # print(id, lm)

                h, w, c = img.shape

                # each point pixel value in hand
                cx, cy = int(lm.x*w) , int(lm.y*h)

                #print (id, cx, cy)

                LandmarkList.append([id, cx, cy])

                # particular point landmark point position
                # if id == 20:
                if Draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED) # if we remove the if part then it would return all of that

        return LandmarkList


def main():
    # check FPS
    PastTime = 0
    CurrentTime = 0

    # for read the webcam video
    capture = cv2.VideoCapture(0)

    Hndetector = HandDetector()

    while True:
        success, img = capture.read()
        img = Hndetector.FindHands(img)
        LandmarkList = Hndetector.FindPosition(img)
        if len(LandmarkList) != 0:
            print(LandmarkList[4])

        #showing the FPS
        CurrentTime = time.time()
        fps = 1/(CurrentTime - PastTime)
        PastTime = CurrentTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("WEBCAM VIDEO", img)
        Key= cv2.waitKey(1)

if __name__ == "__main__":
    main()
