from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
from pydantic import BaseModel
import pytesseract
import numpy as np
import sys
import cv2
import os
import io
import re


def read_img(img):
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    text = pytesseract.image_to_string(img)
    return(text)


app = FastAPI()


class ImageType(BaseModel):
    url: str


@app.post('/ocr')
def prediction(request: Request,file: bytes = File(...)):

    if request.method == 'POST':

        image = io.BytesIO(file)
        image.seek(0)
        img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        label = read_img(frame)
        return label

    return 'SEND A POST REQ.'
