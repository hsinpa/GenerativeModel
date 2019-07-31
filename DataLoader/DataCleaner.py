
from DataLoader.LoaderUtility import LoaderUtility


path1 = "../Dataset/TouhouDataset/Other"
path2 = "../Dataset/TouhouDataset/Marisa"
path3 = "../Dataset/TouhouDataset/Reimu"

valid_images = [".jpg",".png"]
labels = []

loader = LoaderUtility()

imageSetA, labelsA  = loader.GetImageFromPath(path1, valid_images, labels, True)
imageSetB, labelsB  = loader.GetImageFromPath(path1, valid_images, labels, True)
imageSetC, labelsC  = loader.GetImageFromPath(path1, valid_images, labels, True)
