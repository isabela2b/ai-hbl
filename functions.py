import numpy as np
import pandas as pd 
import cv2
import copy
#import keras
import os, shutil

from datetime import datetime
from PyPDF2 import PdfFileWriter, PdfFileReader
#from pdf2image import convert_from_path
#from PIL import Image

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'docx', 'xlsx', 'xls','tiff'])

#poppler_path = r"C:\Program Files\poppler-21.03.0\Library\bin"

def file_name(filename):
    return filename.rsplit('.', 1)[0]

def file_ext(filename):
    return filename.rsplit('.', 1)[1].lower()

def allowed_file(filename):
    return '.' in filename and file_ext(filename) in ALLOWED_EXTENSIONS

def convert_date(timestamp):
    d = datetime.utcfromtimestamp(timestamp)
    formated_date = d.strftime('%d %b %Y %H:%M')
    return formated_date

def img_preprocess(image, image_size):
    im = image.resize((image_size,image_size))
    im = np.array(im)/255
    im = np.expand_dims(im, axis=0)
    return im
