from PIL import Image
from cv2.typing import MatLike, Scalar
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def stripe(img: np.ndarray[int, ...]) -> np.ndarray[int, ...]:
    """Keep only the pixels between two lines y = (r - x cos(theta)) / sin(theta)"""
    output: np.ndarray[int, ...] = np.zeros_like(img)
    lines: MatLike = cv.HoughLines(
        cv.Canny(
            cv.cvtColor(
                img,
                cv.COLOR_BGR2GRAY
            ), 
            50,
            10
        ),
        1,
        np.pi/180,
        240
    )
    # Python wont let me type hind tuple declarations
    r1, theta1 = lines[0][0]
    r2, theta2 = lines[1][0]
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            y1 = (r1 - x * np.cos(theta1)) / np.sin(theta1)
            y2 = (r2 - x * np.cos(theta2)) / np.sin(theta2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            if ymin <= y <= ymax:
                output[y, x] = img[y, x]
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return output

def stripe(img: np.ndarray[int, ...]) -> np.ndarray[int, ...]:
    """Keep only the pixels between two lines y = θx + r"""
    output = np.zeros_like(img) # Copy by value and set to 0
    lines = cv.HoughLines(
        cv.Canny(
            cv.cvtColor(
                img,
                cv.COLOR_BGR2GRAY
            ),
            50,
            10
        ),
        1,
        np.pi/180,
        240
    )
    for line in lines:
        for r, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            it = np.nditer(img, ['multi_index'])
            while not it.finished :
                shape = it.multi_index
                xCoord, yCoord = shape[1], shape[0]
                new = -1
                if >= yCoord >= :
                    new = it[0]
                else:
                    new = 0
                output[shape] = new
            cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    plt.imshow(img)
    return output

def main() -> None:
    """Applies things"""
    image = cv.imread('../data/image31.png')
    angles(image)

if __name__ == "__main__":
    main()

