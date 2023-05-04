import os
import numpy
import cv2
import requests
from SliceImage import create_grids_with_overlap





s = requests.Session()
files = {
    'color': (None, '#00FF00'),
    'mode': (None, 'AI'),
    'type': (None, 'image/jpeg'),
    'X-Value': (None, '100'),
    'X-Operator': (None, '<'),
    #'OriginalSize':"True",
    "Cropped":"false",
    "upscale":"false"


}

headers = {
    # requests won't add a boundary if this header is set when you pass files=
    # 'Content-Type': 'multipart/form-data',
}



WideImagesFolder="/home/zubair/Downloads/flask-master/python-flask-upload-display-image/WideImages/"
Allimages=os.listdir(WideImagesFolder)
for ImageName in Allimages:

    grids=create_grids_with_overlap(WideImagesFolder+ImageName)


    counter=0

    SaveFile = "/home/zubair/Downloads/flask-master/python-flask-upload-display-image/CroppedImages/InferencedCropped/" + ImageName + "/"
    os.mkdir(SaveFile)
    for i in grids:
        OutputImageName="/home/zubair/Downloads/flask-master/python-flask-upload-display-image/CroppedImages/OriginalCropped/"+str(counter)+"__"+ImageName
        print(OutputImageName)
        cv2.imwrite(OutputImageName,i)

        f = open(OutputImageName, "rb")
        im_bytes = f.read()

        response = s.post('http://0.0.0.0:5000/InferenceImage', headers=headers, data=files, files={'image': im_bytes})




        with open(SaveFile +str(counter)+"__"+ImageName, 'wb') as f:
            f.write(response.content)
        counter+=1




