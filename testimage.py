import numpy as np
import cv2


from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
import imagecodecs
from PIL import Image

jpeg = TurboJPEG()

Imagepath="./nav-sprite-global-1x-hm-dsk-reorg.png"
import time
import pyvips

start_time = time.time()
image = cv2.imread(Imagepath, cv2.IMREAD_UNCHANGED)
try:
    for i in range(50):


        success, encoded_image = cv2.imencode(".jpeg", image)
        #success, encoded_image = cv2.imencode(".jpeg", image)

    print("--- %s Cv2seconds ---" % (time.time() - start_time))
except:
    print("opencv Not Supported")
# print(len(encoded_image))


#
# from nvjpeg import NvJpeg
# nj = NvJpeg()
# start_time = time.time()
# for i in range(100):
#     img = nj.read(Imagepath)
#     #jpeg_bytes = nj.encode(image)
#
# # print(len(jpeg_bytes))
#
# print("--- %s NVseconds ---" % (time.time() - start_time))
#
#
# #
# start_time = time.time()
# for i in range(100):
#     in_file = open(Imagepath, 'rb')
#     bgr_array = jpeg.decode(in_file.read(), flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)
#     in_file.close()
#
# # print(len(jpeg_bytes))
#
# print("--- %s decodeseconds ---" % (time.time() - start_time))
#


start_time = time.time()

for i in range(50):
    image=Image.open(Imagepath)
    image=np.array(image)



print(image.shape)
print("--- %s Pillow seconds ---" % (time.time() - start_time))


#
# start_time = time.time()
#
# for i in range(100):
#     with open(Imagepath, 'rb') as fh:
#         encoded = fh.read()
#
#     image = imagecodecs.jpeg_decode(encoded)
# # cv2.imshow("image",image)
# # cv2.waitKey(0)
#
# print("--- %s imagecodecsseconds ---" % (time.time() - start_time))
#
#
start_time = time.time()

for i in range(50):
    data =     image = pyvips.Image.new_from_file(Imagepath)image.write_to_buffer('.png')


# cv2.imshow("image",image)
# cv2.waitKey(0)
print("--- %s pyvipsseconds ---" % (time.time() - start_time))




