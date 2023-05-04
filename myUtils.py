import numpy as np
from PIL import ImageColor
import cv2
import torch
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
from wand.image import Image as WandImage
from wand.display import display as WandDisplay
import pillow_avif
import os
import sys
import traceback
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template,jsonify
import flask
import uuid
import time


jpeg = TurboJPEG()



# def ApplyMask(OriginalImage,pr_mask,MaskColor,Mode):
#
#
#
#     if(not MaskColor==None):
#         MaskColor=ImageColor.getcolor(MaskColor, "RGB")
#         MaskColor=np.array([MaskColor[2],MaskColor[1],MaskColor[0]])
#     else:
#         MaskColor=np.array([0,255,0])
#     if(OriginalImage.shape[2]==4):
#         MaskColor=np.append(MaskColor,255)
#
#     if(Mode=="AI"):
#         OriginalImage[pr_mask] = MaskColor
#     else:
#         OriginalImage[pr_mask] = (OriginalImage[pr_mask] * 0.10) + (MaskColor * 0.90)
#
#
#     return OriginalImage
def ApplyMask(OriginalImage, pr_mask, MaskColor=None, Mode="AI"):
    if MaskColor is not None:
        MaskColor = ImageColor.getcolor(MaskColor, "RGB")
        MaskColor = np.array([MaskColor[2], MaskColor[1], MaskColor[0]], dtype=np.uint8)
    else:
        MaskColor = np.array([0, 255, 0], dtype=np.uint8)

    if OriginalImage.shape[2] == 4:
        MaskColor = np.append(MaskColor, 255)

    if Mode == "AI":
        OriginalImage[pr_mask, :] = MaskColor
    else:
        OriginalImage[pr_mask, :] = (OriginalImage[pr_mask, :] * 0.10) + (MaskColor * 0.90)

    return OriginalImage


def RetransformMask(OriginalWeight,InferenceSize,OriginalHeight,pr_mask,UpScale):
    if(OriginalWeight<InferenceSize and OriginalHeight<InferenceSize and not UpScale):

        StartingHeight=(pr_mask.shape[0]/2)-(OriginalHeight/2)
        EndingHeight=(pr_mask.shape[0]/2)+(OriginalHeight/2)


        StartingWidth=(pr_mask.shape[1]/2)-(OriginalWeight/2)
        EndingWidth=(pr_mask.shape[1]/2)+(OriginalWeight/2)


        pr_mask=pr_mask[int(StartingHeight):int(EndingHeight),int(StartingWidth):int(EndingWidth)]


    else:

        if (OriginalWeight > OriginalHeight):


            TargetWidth = InferenceSize
            targetheight = OriginalHeight / (OriginalWeight / InferenceSize)

            Startingheight = (pr_mask.shape[0] / 2) - (targetheight / 2)
            Endingheight = (pr_mask.shape[0] / 2) + (targetheight / 2)

            pr_mask = pr_mask[int(Startingheight):int(Endingheight), int(0):int(TargetWidth)]


        elif (OriginalHeight > OriginalWeight):

            targetheight = InferenceSize
            TargetWidth = OriginalWeight / (OriginalHeight / InferenceSize)

            StartingWidth = (pr_mask.shape[1] / 2) - (TargetWidth / 2)
            EndingWidth = (pr_mask.shape[1] / 2) + (TargetWidth / 2)



            pr_mask = pr_mask[0:int(targetheight), int(StartingWidth):int(EndingWidth)]

        else:

            targetheight = InferenceSize
            TargetWidth = InferenceSize

            pr_mask = pr_mask[0:int(targetheight), 0:int(TargetWidth)]


    # if(UpScale):
    #     if(OriginalWeight<InferenceSize and OriginalHeight<InferenceSize):
    #         print("CAME HERE")
    #         pr_mask=pr_mask[0:int(targetheight),0:int(TargetWidth)]

    # cv2.imshow('After', pr_mask)
    # cv2.waitKey(0)



    pr_mask = cv2.resize(pr_mask.astype(float), (OriginalWeight, OriginalHeight), interpolation=cv2.INTER_CUBIC)
    pr_mask = np.where(pr_mask < 0.5, 1, 0).astype(bool)

    return pr_mask



def EarlyReturns(X_Operator,MaskArea,Firstnumber,secondnumber):

        ReturnValues=False

        if (not X_Operator == None):

            if(X_Operator=="="):
                if( int(round(MaskArea))==int(Firstnumber)):
                    ReturnValues= True


            elif(X_Operator=="!="):
                if not (int(round(MaskArea))==int(Firstnumber)):
                    ReturnValues= True

            elif(X_Operator=="<"):
                if not (int(round(MaskArea))<int(Firstnumber)):
                    ReturnValues= True

            elif(X_Operator=="<="):
                if not (int(round(MaskArea))<=int(Firstnumber)):
                    ReturnValues= True


            elif(X_Operator==">"):
                if not (int(round(MaskArea))>int(Firstnumber)):
                    ReturnValues= True

            elif(X_Operator==">="):
                if not (int(round(MaskArea))>=int(Firstnumber)):
                    ReturnValues= True

            elif(X_Operator=="bt"):
                print("CAME HERE")
                if not (int(secondnumber)>int(round(MaskArea))>int(Firstnumber)):
                    ReturnValues= True

            elif(X_Operator=="!bw"):
                if (int(secondnumber)>int(round(MaskArea))>int(Firstnumber)):
                    ReturnValues= True

        return ReturnValues


def CalculateMaskArea(pr_mask):
    non_zero_pixels = np.count_nonzero(pr_mask)
    total_pixels = pr_mask.size
    mask_area = (non_zero_pixels / total_pixels) * 100

    return mask_area








def InferenceImage(NormalizedImages,OriginalSizeFlag,model_trt,best_model,DEVICE):
    x_tensor=torch.stack(NormalizedImages).to(DEVICE)

    print(x_tensor.shape)
    print("CEHCING THIS")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    print(OriginalSizeFlag , " TENSORRRTT")

    if ( not OriginalSizeFlag):

        pr_mask = model_trt(x_tensor)

    else:
        with torch.no_grad():
            print("PREDICT SIZE:  " , x_tensor.shape)
            pr_mask = best_model.predict(x_tensor)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print("INFERENCE TIME:",elapsed)

    pr_masks = pr_mask.sigmoid()

    pr_masks = (pr_masks < 0.5).float()
    pr_mask = pr_masks.squeeze().cpu().numpy()

    return pr_mask


def Preprocess(OriginalImage,image,UpScale,augmentationUpScaled,augmentation,img_transforms,effnet):

    if(OriginalImage.shape[2]==4):
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)



    if(UpScale):
        sample = augmentationUpScaled(image=image)
        image = sample['image']
    else:

        sample = augmentation(image=image)
        image= sample['image']


    if(effnet):
        NormalizedImage = img_transforms(image)

        return NormalizedImage
    else:
        return image





def ReadImages(FileExtension,imagename,FileType):
    try:
        if (FileExtension == ".jpeg"):



            in_file = open(imagename, 'rb')
            img = jpeg.decode(in_file.read(), flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)

            # img = cv2.imread(imagename)
            #
            #
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)





        elif (FileExtension == ".png" or FileExtension == ".webp"):
            print("Reading png or WEBP using opencv")
            img = cv2.imread(imagename, cv2.IMREAD_UNCHANGED)
            if (img.shape[2] == 4):
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("CHECKING HERE123   ", img.shape)

        else:  ### other formats apart from jpeg ,webp and png
            print("RAISING")

            raise ValueError('Pillow Format')

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as E:

        try:
            print("FROM HERE123")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(E)

            start_time = time.time()
            imgO = Image.open(imagename)
            img = np.array(imgO)

            if (imgO.format == "PNG" and imgO.mode == "P" and len(img.shape) < 3):
                imgO = imgO.convert("RGBA")
                img = np.array(imgO)

            if (len(img.shape) < 3):
                OpenedImage = imgO.convert("RGB")

                img = np.array(OpenedImage)

        except:

            print("CANT READ FROM Pillow")
            try:
                img = WandImage(filename=imagename)

                print(np.unique(img))

                WandDisplay(img)
                img = np.array(img)

            except Exception as E:
                print(E)

                print("CANT EVEN READ FROM WAND")
                print("--------------------------------------------------------------------------")
                print('----------------------------------------------------------------')
                print(str(uuid.uuid4()) + FileExtension)
                # file.save("./SaveImages/"+str(uuid.uuid4())+FileExtension)

                resp = flask.Response()  # ,mimetype=file.content_type

                resp.headers["X-Coverage"] = 0
                # resp.headers["X-latency"] = OutputTime
                resp.headers["content-type"] = FileType

                return resp

    return img





def ParseParams(request):
    OriginalSize = request.get('OriginalSize', "true")

    if (OriginalSize.lower() == "false"):
        OriginalSizeFlag = False
    else:
        OriginalSizeFlag = True

    MaskColor = request.get('color', None)

    Mode = request.get('mode', "FI")

    X_Value = request.get("X-Value", 100)

    X_Operator = request.get("X-Operator", "<")

    InfSize = request.get("infsize", 640)
    InfSize = int(InfSize)

    FileType = request.get("type", "image/jpeg")

    CroppedVersion = request.get('Cropped', True)

    if (type(CroppedVersion) != bool):
        if (CroppedVersion.lower() == "false"):
            CroppedVersion = False
        else:
            CroppedVersion = True

    upscale = request.get('upscale', "false")
    if (upscale.lower() == "false"):
        upscale = False
    else:
        upscale = True
    Firstnumber = None
    secondnumber = None

    if (X_Operator == "bt" or X_Operator == "!bw"):

        Firstnumber = int(X_Value.split(" ")[0])
        secondnumber = int(X_Value.split(" ")[1])
    else:
        Firstnumber = int(X_Value)

    Sliced = request.get("Sliced", True)
    if (Sliced.lower() == "false"):
        Sliced = False
    else:
        Sliced = True


    Effnet = request.get("Effnet", False)
    if (Effnet.lower() == "false"):
        Effnet = False
    else:
        Effnet = True

    Tensorrt=request.get("tensorrt", False)
    if (Tensorrt.lower() == "false"):
        Tensorrt = False
    else:
        Tensorrt = True




    return OriginalSizeFlag,MaskColor,Mode,X_Value,X_Operator,InfSize,FileType,CroppedVersion,upscale,Firstnumber,secondnumber,Sliced,Effnet,Tensorrt


