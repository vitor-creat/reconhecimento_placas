import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--img_path', default="/home/vitor/Documentos/CNN_letters_custom/P/aug10384_3.jpg",type=str)
args = parser.parse_args()
def detection_plate():

    detection_model = YOLO(Path(".","ctc_placas","detection","best.pt"))
    image_path = args.img_path
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detection_model(image)

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:

            x1, y1, x2, y2 = map(int, boxes[0])

            padding = 5

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)

            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            roi = img_rgb[y1:y2, x1: x2]

            # Convert to grayscale for better visualization of characters
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Invert back to black characters on white background
            roi_thresh = cv2.bitwise_not(roi_thresh)

            # Display the result
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_rgb)
            plt.title('Imagem Original')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(roi_thresh, cmap='gray')
            plt.title('Placa Processada')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(roi)
            plt.title('Região de Interesse')
            plt.axis('off')

            plt.show()

            # Save processed image
            output_path = "processed_license_plate.png"
            cv2.imwrite(output_path, roi_thresh)

            print(f"Processed license plate saved as {output_path}")

            return roi
        

if __name__ == "__main__":
    detection_plate()