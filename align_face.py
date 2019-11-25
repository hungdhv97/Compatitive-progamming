from imutils.face_utils import FaceAligner
from PIL import Image
import numpy as np
import imutils 
import dlib
import cv2

# Tạo bộ nhận diện khuôn mặt dlib
detector = dlib.get_frontal_face_detector()
# Tạo bộ dự đoán phân vùng trên khuôn mặt
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Nhận diện khuôn mặt
fa = FaceAligner(predictor, desiredFaceWidth=256, desiredLeftEye=(0.371, 0.480))

# Đầu vào:Dãy numpy ảnh với 3 kênh RGB
# Đầu ra: (Dãy numpy, Tìm đc mặt(True, False))
def align_face(img):
	img = img[:, :, ::-1] #Chuyển RGB sang BGR
	img = imutils.resize(img, width=800)

	# Nhận diện khuôn mặt với ảnh xám
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)

	if len(rects) > 0:
		# Nhận diện khuôn mặt với các vị trí trên mặt
		align_img = fa.align(img, gray, rects[0])[:, :, ::-1]
		align_img = np.array(Image.fromarray(align_img).convert('RGB'))
		return align_img, True
	else:
		# Không tìm được khuôn mặt
		return None, False

# Đầu vào: đường dẫn ảnh
# Đầu ra: aligned_img
def align(img_path):
	img = Image.open(img_path)
	img = img.convert('RGB')
	img = np.array(img)
	x, face_found = align_face(img)
	return x