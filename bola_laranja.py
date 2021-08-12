import sys
import numpy as np
import cv2 as cv

img = cv.imread("fotos/foto1.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_range = np.array([5, 100, 20])
upper_range = np.array([25, 255, 255])

mask = cv.inRange(hsv, lower_range, upper_range)
# cv.imshow("orange", mask)
cv.imwrite("laranja2masked.jpg", mask)

def main(argv):
    
    default_file = 'laranja2masked.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    
    gray = cv.medianBlur(gray, 5)
    
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=60, param2=20,
                               minRadius=100, maxRadius=300)
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            print(radius)
            cv.circle(src, center, radius, (255, 0, 255), 3)
    
    
    cv.imshow("detected circles", src)
    cv.waitKey(0)
    cv.imwrite("resultado.jpg", src)
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])