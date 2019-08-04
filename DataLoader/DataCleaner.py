
from DataLoader.LoaderUtility import LoaderUtility
from typing import List
import os, os.path


path1 = "../Dataset/TouhouDataset/Other/"
path2 = "../Dataset/TouhouDataset/Marisa/"
path3 = "../Dataset/TouhouDataset/Reimu/"
resizeFolder = "../Dataset/TouhouDataset/ResizeFolder/"

valid_images = [".jpg",".png"]
labels = []


def GetImageFromPath(p_path: str, p_valid_formats: List[str]):
    path = []

    for f in os.listdir(p_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in p_valid_formats:
            continue

        path.append(p_path + f)

    return path

loader = LoaderUtility()

imageSetA = GetImageFromPath(path1, valid_images)
imageSetB = GetImageFromPath(path2, valid_images)
imageSetC = GetImageFromPath(path3, valid_images)
allImageArray = imageSetA + imageSetB + imageSetC
print(allImageArray)

for path in allImageArray:
    lastIndex = path.rindex("/") + 1
    newFileName = resizeFolder + path[lastIndex:-3] + "jpeg"
    loader.resize_canvas(path, newFileName, img_type="jpeg", canvas_width=256, canvas_height=256)