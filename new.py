import cv2
import argparse
import dlib
import time

# handle command line arguments

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='D:\Big data\Computer Vision\ImageDetection\FacerecogPython')

ap.add_argument('-w', '--weights', default='.\mmod_human_face_detector.dat', help='D:\Big data\Computer Vision\ImageDetection\FacerecogPython')

args = ap.parse_args()

image = cv2.imread(args.image)

if image is None:
    print('could not read input image')
    exit()
else:
    print("found obama image")

hog_face_detector = dlib.get_frontal_face_detector()

cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

start = time.time()

# apply face detection (hog)
faces_hog = hog_face_detector(image, 1)

end = time.time()

print("Execution Time (in seconds) :")
print("HOG : ", format(end - start, '.2f'))

# loop over detected faces
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

# apply cnn for face detection

start = time.time()

faces_cnn = cnn_face_detector(image, 1)

end = time.time()

print("CNN : ", format(end - start, '.2f'))

# loop over detected faces

for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

    # draw box over face
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)

# write at the top left corner of the image
# for color identification
img_height, img_width = image.shape[:2]
cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)

# display output image
cv2.imshow("face detection with dlib", image)
cv2.waitKey()

# save output image
cv2.imwrite("cnn_face_detection.png", image)

# close all windows
cv2.destroyAllWindows()
