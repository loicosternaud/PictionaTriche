import cv2 as cv
import numpy as np
import gradio as gr

class Shape:
    def __init__(self, img):
        self.img = img

    def detect(self, contour):
        shape = "undefined"
        epsilon = 0.03 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            shape = "Square"
        elif len(approx) > 5 and len(approx) < 15:
            shape = "triangle"
        else:
            shape = "Circle"
        return shape

def draw_and_detect(img):
    img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
    img_object = Shape(img)
    threshold, contours = preprocessing_image(img)
    detected_shapes = [img_object.detect(contour) for contour in contours]
    return ', '.join(detected_shapes)

def preprocessing_image(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(img_gray, 127, 255, 0)
    kernel = np.ones((5, 5), np.uint8)
    cv.dilate(threshold, kernel, iterations=1)
    threshold = cv.GaussianBlur(threshold, (15, 15), 0)
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return threshold, contours

iface = gr.Interface(fn=draw_and_detect, inputs="sketchpad", outputs="text", live=True)
iface.launch()