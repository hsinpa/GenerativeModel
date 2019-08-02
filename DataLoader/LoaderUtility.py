import os, os.path

import cv2
import numpy as np
from PIL import Image
import math
from typing import List

class LoaderUtility:

    def GetLabelIndexFromImages(self, image_name: str, labels):
        label_s_index = image_name.rfind("_") + 1

        label = image_name[label_s_index:-4]
        return labels.index(label)

    def GetImageFromPath(self, p_path: str, p_valid_formats: List[str], p_define_label: List[str], p_normalized: bool):
        imgs = []
        labels = []

        for f in os.listdir(p_path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in p_valid_formats:
                continue

            #Append Labels only if its exist
            if (len(p_define_label) > 0):
                labels.append(self.GetLabelIndexFromImages(f, p_define_label))

            #Get Grayscale images
            image = cv2.imread(p_path+f, cv2.IMREAD_GRAYSCALE)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if p_normalized:
                image = image / 255

            imgs.append(image)
        return np.asarray(imgs), np.asarray(labels)

    def shuffle(self, x, y):
        idx = np.random.permutation(len(x))
        return x[idx], y[idx]

    def RewriteImagePath(self, path, label, image):
        fileCount = len( os.listdir(path))
        fileName = "{}/gesture_{}_{}.png".format(path, fileCount, label)
        cv2.imwrite(fileName, image)

    def FlipImage(self, raw_image):
        newImgs = []

        newImgs.append(cv2.flip(raw_image, 1))
        newImgs.append(cv2.flip(raw_image, 0))
        newImgs.append(cv2.flip(raw_image, -1))
        return newImgs


    def resize_canvas(self, old_image_path : str ="314.jpg", new_image_path : str ="save.jpg",
                      canvas_width : int =500, canvas_height : int =500):
        """
        Resize the canvas of old_image_path.

        Store the new image in new_image_path. Center the image on the new canvas.

        Parameters
        ----------
        old_image_path : str
        new_image_path : str
        canvas_width : int
        canvas_height : int
        """
        im = Image.open(old_image_path)
        old_width, old_height = im.size

        # Center the image
        x1 = int(math.floor((canvas_width - old_width) / 2))
        y1 = int(math.floor((canvas_height - old_height) / 2))

        mode = im.mode
        if len(mode) == 1:  # L, 1
            new_background = (255)
        if len(mode) == 3:  # RGB
            new_background = (255, 255, 255)
        if len(mode) == 4:  # RGBA, CMYK
            new_background = (255, 255, 255, 255)

        newImage = Image.new(mode, (canvas_width, canvas_height), new_background)
        newImage.paste(im, (x1, y1, x1 + old_width, y1 + old_height))

        if (old_height > canvas_height or old_width > canvas_width or ( abs(old_height - old_width) < 50)):
            wpercent = (canvas_width / old_width)
            hsize = int((float(old_height) * float(wpercent)))
            newImage = im.resize((canvas_width, hsize), Image.ANTIALIAS)

        newImage.save(new_image_path)