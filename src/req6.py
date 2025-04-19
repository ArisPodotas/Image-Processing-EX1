from PIL import Image
from cv2.typing import Scalar
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def edges(img: np.ndarray[int, ...], kernel: np.ndarray[int, ...]) -> np.ndarray[int, ...]:
    """Isolates the edges with a laplacian matrix"""
    #laplacian
    blur: np.ndarray = cv.GaussianBlur(img,(3,3),0)
    output: np.ndarray = cv.filter2D(blur, -1, kernel)
    return output

def corners(img: np.ndarray) -> None:
    """Isolates the corners of the image"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
    if corners is not None:
        corners = np.intp(corners)
        for i in corners:
            x, y = i.ravel()
            print(f"Corner at: ({x}, {y})")
            cv.circle(img, (x, y), 3, (0, 255, 0), 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Corners Detected")
    plt.axis('off')
    plt.show()

def angles(img: np.ndarray[int, ...]) -> None:
    """Will find the angles of the lines in the image"""
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
            cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)
            print(f'r: {r}, theta: {theta}')
    plt.imshow(img)

def main() -> None:
    """Applies things"""
    # matrix = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    image = cv.imread('../data/image11.jpg')
    # output: np.ndarray = edges(image, matrix)
    # Image.fromarray(output).save('../figures/Image_11_edges.jpg')
    output: np.ndarray = edges(image, kernel)
    Image.fromarray(output).save('../figures/Image_11_edges_rough.jpg')
    angles(image)
    corners(image)

if __name__ == "__main__":
    main()

