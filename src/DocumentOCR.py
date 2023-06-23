from OCR.detection.text_detector import PaddleTextDetector
from OCR.recognition.text_recognizer  import TextRecognizer_TMA
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from skimage.metrics import structural_similarity as ssim
from sklearn import preprocessing


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	#t = np.sum((imageA.astype("float"))**2)*np.sum((imageB.astype("float"))**2)
	#err = err/(t**(1/2))
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def correlation(imageA, imageB):
	err = np.sum(imageA.astype("float") * imageB.astype("float"))
	#err /= float(imageA.shape[0] * imageA.shape[1])
	t = np.sum((imageA.astype("float"))**2) * np.sum((imageB.astype("float"))**2)
	err = err/(t**(1/2))
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
	
def correlation_coeficient(imageA, imageB):
	h = imageA.shape[0]
	w = imageA.shape[1]

	t1 = imageA.astype("float") - (1/(w*h))*sum(imageA.astype("float"))
	t2 = imageB.astype("float") - (1/(w*h))*sum(imageB.astype("float"))
	err = np.sum(t1 * t2)
	#err /= float(imageA.shape[0] * imageA.shape[1])
	t = np.sum(t1**2) * np.sum(t2**2)
	err = err/(t**(1/2))
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


if __name__ == '__main__':
    print("ok")
    n = 2
    temp_img = cv2.imread('src/imgpsh_fullsize_anim.jpeg',cv2.IMREAD_COLOR)
    img = cv2.imread('src/imgpsh_fullsize_anim_test.jpeg',cv2.IMREAD_COLOR)
    
    print(img.shape)
    detector = PaddleTextDetector.getInstance()
    recognizer = TextRecognizer_TMA()
    det = detector.detect(img)
    print(det.shape)
    text_list = []
    box_list = []
    count = 0
    
    for box in tqdm(det):
        box = np.array(box).astype(np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [box], (255, 255, 255))
        masked_image = cv2.bitwise_and(img, mask)
        x, y, w, h = cv2.boundingRect(box)
        text_img = masked_image[y:y+h, x:x+w]
   
        if text_img.shape[0] != 0 and text_img.shape[1] != 0:
            text,time = recognizer.recognize([text_img])
            text_list.append(text[0][0])
            box_list.append(box)
        else:
            print(text_img.shape)
            continue
    print(box_list[100])
    print(text_list[100])
    #np.save("src/OCR/recognition/results/template_position.npy", box_list, allow_pickle=True, fix_imports=True)
    
    '''
    f = open("src/OCR/recognition/results/demofile2.txt", "a")
    f.write(' '.join(text_list))
    f.close()
    '''
    isClosed = True
    color = (250, 253, 15)
    # Line thickness of 2 px
    thickness = 2
	

    '''
    for i in range(len(text_list)):
       box = np.array(box_list[i]).astype(np.int32).reshape(-1, 2)
       #cv2.polylines(img, [box], True, color=(0, 255, 0), thickness=3)
       #cv2.putText(img, str(text_list[i]),(box[1][0],box[1][1]),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
       #if(i==100):
       	#print(box[0][1])
       
       img = cv2.polylines(img, [box], isClosed, color, thickness) 
    
    cv2.imwrite("src/OCR/recognition/results/img.jpg", img)

    temp_img = four_point_transform(img, box_list[247])
    cv2.imwrite("src/OCR/recognition/results/temp_img_.jpg",temp_img)
    print(temp_img.shape)
    
    '''
    temp_box_matrix = np.load("src/OCR/recognition/results/template_position.npy")
    print(temp_box_matrix.shape)
    box_matrix = np.array(box_list)
    print(box_matrix.shape)
    ND_list = []
    er_list = []
    min_list = []
    temp_po = temp_box_matrix[248]
    x = four_point_transform(temp_img,temp_po)
    
    temp_X = [temp_po[0][0], temp_po[1][0], temp_po[2][0], temp_po[3][0]]
    temp_Y = [temp_po[0][1], temp_po[1][1], temp_po[2][1], temp_po[3][1]]
    td_temp_X = (min(temp_X)+max(temp_X)/2).astype('int')
    td_temp_Y = (min(temp_Y)+max(temp_Y)/2).astype('int')
    for i in range(248):
    	k = box_matrix[i]
    	y = four_point_transform(img,k)
    	p_x = cv2.resize(x,(max(x.shape[0],y.shape[0]),max(x.shape[1],y.shape[1])), cv2.INTER_AREA)
    	p_y = cv2.resize(y,(max(x.shape[0],y.shape[0]),max(x.shape[1],y.shape[1])), cv2.INTER_AREA)
    	p = mse(p_x,p_y)
    	X = [k[0][0], k[1][0], k[2][0], k[3][0]]
    	Y = [k[0][1], k[1][1], k[2][1], k[3][1]]
    	td_X = (min(X)+max(X)/2).astype('int')
    	td_Y = (min(Y)+max(Y)/2).astype('int')
    	d = np.sqrt((td_X-td_temp_X)**2+(td_Y-td_temp_Y)**2)
    	min_list.append(d)
    	er_list.append(p)
    dis_arr = np.array(min_list)
    dis_arr = (dis_arr-min(dis_arr))/(max(dis_arr)-min(dis_arr))
    er_arr = np.array(er_list)
    er_arr = (er_arr-min(er_arr))/(max(er_arr)-min(er_arr))
    result = dis_arr + er_arr
    print(np.argmin(result))
    print(result[np.argmin(result)])
    print(er_arr[np.argmin(result)])
    print(dis_arr[np.argmin(result)])
    print(result[245])
    print(er_arr[245])
    print(dis_arr[245]) 
 		  
     
    m = four_point_transform(img, box_matrix[241])
    cv2.imwrite("src/OCR/recognition/results/y_img_.jpg",m)
    
    n = four_point_transform(temp_img, temp_box_matrix[248])
    cv2.imwrite("src/OCR/recognition/results/x_img_.jpg",n)

