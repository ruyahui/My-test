# USAGE
# python face.py --encodings encodings.pickle -d hog

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import time
import threading

# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
        
    # 攝影機連接。
        self.capture = cv2.VideoCapture(0)

    def start(self):
    # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
    # 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
    # 當有需要影像時，再回傳最新的影像。
        return self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
RTSP= "rtsp://admin:888888@192.168.11.13:10554/tcp/av0_0"
# 連接攝影機
video_capture = ipcamCapture(RTSP)

# 啟動子執行緒
video_capture.start()
# 暫停1秒，確保影像已經填充
time.sleep(1)
matchid=0
while True:
# load the input image and convert it from BGR to RGB
	image =video_capture.getframe()
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	tStart = time.time()
# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	print("[INFO] recognizing faces...0")
	encodings = face_recognition.face_encodings(rgb, boxes)
#	tEnd = time.time()
#	print ("It cost %f sec" % (tEnd - tStart))
#	tStart = time.time()
# initialize the list of names for each face detected
	names = []
	print("[INFO] recognizing faces...1")
# loop over the facial embeddings
	for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding,tolerance=0.4)

		name = "Unknown"
		print("[INFO] recognizing faces...2")
		# check to see if we have found a match
		if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			print("[INFO] recognizing faces...3")
		# loop over the matched indexes and maintain a count for
		# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				matchid=i
				counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		# update the list of names
		names.append(name)

	tEnd = time.time()
	print ("It cost %f sec" % (tEnd - tStart))
	if name != "Unknown":
		distance = face_recognition.face_distance(data["encodings"][matchid], encodings)	
		print("Distance:",'%0.2f' % distance)
# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name+str('%0.2f' % distance), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

# show the output image
	#cv2.waitKey(0)
	cv2.startWindowThread()	
	cv2.imshow("Image", image)
	if cv2.waitKey(100) == 27:
            break

