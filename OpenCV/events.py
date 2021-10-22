import cv2 as cv
import numpy as np

#List all events
all_events = [i for i in dir(cv) if 'EVENT' in i]
print( all_events )
