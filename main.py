import os
import uuid
import sys
from app import app
from flask import  request, redirect, url_for, render_template,jsonify
from Engine import RunImage
from PIL import Image
from filemime import filemime
import io
import flask
import base64
import time
import numpy as np
import cv2
import traceback
from myUtils import ReadImages,ParseParams,ApplyMask,CalculateMaskArea,EarlyReturns

from turbojpeg import TurboJPEG

fileObj = filemime()
MaskAreas = {}
Inferencetimes = {}
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','webp',"gif","ico","icns"])
import hashlib

jpeg = TurboJPEG()

from SliceImage import create_grids_with_overlap




@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response



counter=0
TotalImages = 0

import magic





@app.route('/InferenceImage', methods=['POST'])
def InferenceImage():
    global counter,TotalImages


    Beginningtime=time.time()
    counter+=1



    start_timeStart = time.time()
    Timings={}
    try:


        file = request.files['image']

        print("BEFORE INFERENCE111:  ", time.time() - Beginningtime)

        OriginalSizeFlag,MaskColor,Mode,X_Value,X_Operator,InfSize,FileType,CroppedVersion,upscale,Firstnumber,secondnumber,Sliced,Effnet,tensorrt=ParseParams(request.form)




        if(X_Operator == None or X_Operator == None):
            resp = flask.Response("X-Value or X-Operator Not Provided" , status=500)  # ,mimetype=file.content_type
            return resp

        FileExtension = "." + FileType.split("/")[-1]

        imagename="./SaveImages/" + str(uuid.uuid4()) + FileExtension
        print(imagename + "   " +str(counter))

        file.save(imagename)


        Md5=hashlib.md5(open(imagename, 'rb').read()).hexdigest()


        mime= magic.from_file(imagename, mime=True)
        FileExtension="."+mime.split("/")[-1]
        if(FileExtension==None):



            FileExtension=".png"




        img=ReadImages(FileExtension,imagename,FileType)
        if(not type(img)==np.ndarray):
            return img



        if (img.shape[0] < 25 and img.shape[1] < 25):
            resp = flask.Response()  # ,mimetype=file.content_type
            resp.headers["X-Coverage"] = 0.0
            resp.headers["content-type"] = FileType
            resp.headers["md5"] = Md5
            return resp


        print("BEFORE INFERENCE4:  ",time.time()-Beginningtime)

        img0 = img.copy()

        CroppedImages=[]
        CroppedCoordinates=[]

        EachHeight=[]
        EachWidth=[]


        if(img0.shape[0]<400 and img0.shape[1] <400 ):

            upscale=True
            CroppedVersion = False

        CroppedVersion=False

        # print("NUMBER OF CROPS:  ",str(len(CroppedImages)))


        print("BEFORE INFERENCE5:  ",time.time()-Beginningtime)

        try:
            if(not CroppedVersion):
                if(True): #len(CroppedImages)>0

                    InferenceTimings=time.time()

                    if(not Sliced):
                        added_image, MaskArea, EarlyReturn,PureMask = RunImage(
                            OriginalSizeFlag, MaskColor, Mode, image=[img0],
                            X_Operator=X_Operator, Firstnumber=Firstnumber, secondnumber=secondnumber, UpScale=upscale,InferenceSize=InfSize,Effnet=Effnet,tensorrt=tensorrt)

                        added_image = added_image[0]
                        MaskArea = MaskArea[0]
                        print("THIS IS MASK AREA:  ", MaskArea)
                        EarlyReturn = EarlyReturn[0]
                        print("Inference Times:  ", time.time() - InferenceTimings)

                    else:
                        GridTime=time.time()
                        SlicedImages, GridLocations = create_grids_with_overlap(img0)
                        print("GRID MAKING TIME:  ",time.time()-GridTime)
                        added_image, MaskArea, EarlyReturn,PureMask = RunImage(
                            OriginalSizeFlag, MaskColor, Mode, image=SlicedImages,
                            X_Operator=X_Operator, Firstnumber=Firstnumber, secondnumber=secondnumber, UpScale=upscale,InferenceSize=InfSize,Effnet=Effnet,tensorrt=tensorrt)

                        FinalMask = np.zeros((img0.shape[0], img0.shape[1]))

                        print("Inference Times:  ",time.time()-InferenceTimings)




                        for EachInference in range(len(PureMask)):
                            c1, c2 = GridLocations[EachInference]


                            FinalMask[c1[0]:c1[1], c2[0]:c2[1]] += PureMask[EachInference]

                        FinalMask = (FinalMask >= 1)

                        img0 = ApplyMask(img0, FinalMask, MaskColor, Mode)

                        MaskArea = CalculateMaskArea(FinalMask)
                        Return = EarlyReturns(X_Operator, MaskArea, Firstnumber, secondnumber)

                        # ApplyMask()

                        EarlyReturn = Return
                        added_image = img0



                else:
                    print("NO NEED TO RUN Segmenter")
                    added_image, MaskArea, EarlyReturn=img0,0,False


            else:


                if(len(EachHeight)>0):

                    AverageHeight= int(sum(EachHeight)/len(EachHeight))
                    AverageWidth= int(sum(EachWidth)/len(EachWidth))

                    LargestHeight=max(EachHeight)
                    LargestWidth=max(EachWidth)
                    AverageDim=int((AverageWidth+AverageHeight)/2)
                    AverageDim=int(AverageDim*(2/3))
                    AverageDim=AverageDim- (AverageDim%32)+32

                    if(AverageDim<192):
                        AverageDim=192
                    if(AverageDim>512):
                        AverageDim=512


                    if(LargestWidth<512 and LargestHeight < 512):
                        AverageDim=max(EachHeight + EachWidth)
                        AverageDim=AverageDim-(AverageDim%32)+32





                    added_image, MaskArea, EarlyReturn,PureMask = RunImage(
                        OriginalSizeFlag, MaskColor, Mode, image=CroppedImages,
                        X_Operator=X_Operator, Firstnumber=Firstnumber, secondnumber=secondnumber, UpScale=upscale,
                        InferenceSize=AverageDim)

                    FinalMask=np.zeros((img0.shape[0],img0.shape[1]))


                    for EachInference in range(len(PureMask)):
                        c1, c2=CroppedCoordinates[EachInference]

                        FinalMask[c1[1]:c2[1], c1[0]:c2[0]]+=PureMask[EachInference]

                    FinalMask = (FinalMask >= 1)



                    img0=ApplyMask(img0,FinalMask,MaskColor,Mode)

                    MaskArea=CalculateMaskArea(FinalMask)
                    Return = EarlyReturns(X_Operator, MaskArea, Firstnumber, secondnumber)



                    #ApplyMask()

                    EarlyReturn=Return
                    added_image=img0
                else:
                    EarlyReturn = True
                    MaskArea = 0.0
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            print(e)
            print("PROBLEM ON THIS IMAGE:  ",img0.shape)
            cv2.imwrite("./BatchResults/"+"OriginalProblem"+str(counter)+".png",cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))





        SENDINGDATA=time.time()




        if(EarlyReturn):
            resp = flask.Response()  # ,mimetype=file.content_type


            resp.headers["X-Coverage"] = 0 #MaskArea
            resp.headers["X-latency"] = 80.0
            resp.headers["content-type"] = FileType


            for i in Timings.keys():
                resp.headers[i] = str(float(Timings[i]) * 1000)

            resp.headers["Total_Time_in_Code"]=(time.time()- start_timeStart) *1000
            resp.headers["ImageDir"] = imagename
            resp.headers["md5"] = Md5
            return resp



        masked_img = added_image


        start_time = time.time()

        if (masked_img.shape[2] == 3):
            FileExtension=".jpeg"
        if(masked_img.shape[2]==4):
            FileExtension=".png"


        # cv2.imwrite("./BatchResults/"+"Original"+str(TotalImages)+".png",cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        TotalImages+=1



        #Timings["BeforeEncodeImage"] = time.time() - start_time
        try:

            if (masked_img.shape[2] == 4):
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGRA2RGBA)

                success, encoded_image = cv2.imencode(FileExtension, masked_img)

                enc_img = encoded_image.tobytes()


            else:
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

                enc_img=jpeg.encode(masked_img, quality=70)




        except Exception as E:
            print(E , " THIS IS THE PROBLEM")
            # success, encoded_image = cv2.imencode(".png", masked_img)
            # enc_img = encoded_image.tobytes()
            print("EXCEPTION")
            if (masked_img.shape[2] == 4):
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGRA2RGBA)

            else:
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            masked_img = Image.fromarray(masked_img)



            buf = io.BytesIO()
            masked_img.save(buf, format=FileExtension[1:])
            enc_img = buf.getvalue()


        resp = flask.Response(enc_img)#,mimetype=file.content_type
        resp.headers["X-Coverage"] = MaskArea
        resp.headers["content-type"]=FileType
        resp.headers["ImageDir"]=imagename
        resp.headers["md5"]=Md5


        print("TOTAL TIME:" , time.time()-start_timeStart)

        print("Sendingtime:  ",time.time()-SENDINGDATA)

        return resp

    except Exception as E:
        tb = traceback.format_exc()
        print(tb)
        print(E)
        resp = flask.Response("ERROR on file:  "+file.filename,status=500)#,mimetype=file.content_type
        return resp



        print("ERROR:  ",E)



@app.route('/InferenceImageBatch', methods=['POST'])
def InferenceImageBatch():
        print("CHECKING HEADERS")
        print(request.headers)



        global counter

        counter += 1

        JsonObject = request.json

        OriginalSizeFlag,MaskColor,Mode,X_Value,X_Operator,InfSize,FileType,CroppedVersion,upscale,Firstnumber,secondnumber=ParseParams(JsonObject)


        ImageArray=JsonObject['images']

        NewResponseImageArrayAll=[]

        AllImagesMerged=[]

        ImageCounter=0




        for EachImageObject in ImageArray:
            try:

                NewResponseImageArray={}



                ImageID=EachImageObject['id']

                ImageBase=EachImageObject['b']

                if(not ImageBase[-2:]=="=="):

                    nparr = np.fromstring(base64.b64decode(ImageBase+ "=="), np.uint8)
                else:
                    nparr = np.fromstring(base64.b64decode(ImageBase), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                print(img.shape , " CHECKING THIS123")

                if (img.shape[0] < 25 and img.shape[1] < 25):

                    NewResponseImageArray['id']=ImageID
                    NewResponseImageArray['b']=None
                    NewResponseImageArrayAll.append(NewResponseImageArray)
                    continue


                if (img.shape[2] == 4):
                    Uniques = np.unique(img[:, :, 3])
                    if (len(Uniques) == 1 and Uniques[0] == 255):
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        print("IMAGE CHANGED - 123")


                if (img.shape[2] == 4):
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                NewResponseImageArray['id'] = ImageID
                NewResponseImageArray['b'] = img
                NewResponseImageArray['X-Coverage'] = None
                NewResponseImageArrayAll.append(NewResponseImageArray)
                AllImagesMerged.append(img)

            except Exception as E:
                print(traceback.format_exc())
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print("CAME IN EXCEPTION")
                NewResponseImageArray['id'] = ImageID
                NewResponseImageArray['b'] = None
                NewResponseImageArray['X-Coverage'] = None
                NewResponseImageArrayAll.append(NewResponseImageArray)



        # img0 = img.copy()




        try:
            added_image, MaskArea,  EarlyReturn,PureMask = RunImage(
                OriginalSizeFlag, MaskColor, Mode, image=AllImagesMerged,
                X_Operator=X_Operator, Firstnumber=Firstnumber, secondnumber=secondnumber, UpScale=upscale,InferenceSize=256)
        except:
            print("THIS WAS THE CRASH")
            #print(AllImagesMerged)
            print(ImageArray)



        CurrentCounter=0





        for i in range(len(NewResponseImageArrayAll)):


            if(NewResponseImageArrayAll[i]['b'] is None):
                print("THIS IMAGE IS BROKEN")
                continue


            masked_img=added_image[CurrentCounter]


            if (masked_img.shape[2] == 4):
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGRA2RGBA)
                string = base64.b64encode(cv2.imencode('.png', masked_img)[1]).decode()

            else:
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

                # cv2.imwrite("./YoloResults/"+str(ImageCounter)+".jpeg",
                #             masked_img)

                string = base64.b64encode(cv2.imencode('.jpg', masked_img)[1]).decode()


            NewResponseImageArrayAll[i]['b'] = string
            NewResponseImageArrayAll[i]["X-Coverage"] = MaskArea[CurrentCounter]



            ImageCounter+=1
            CurrentCounter+=1







        return jsonify(NewResponseImageArrayAll)










    #return flask.jsonify(response_data )





# return redirect(url_for('static', filename='Outputimages/' + filename), code=301)
# OriginalSizeFlag = request.form.get('hello')
# RunImage(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename, OriginalSizeFlag)
from werkzeug.serving import WSGIRequestHandler
if __name__ == "__main__":
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host='0.0.0.0', threaded=True,port=5000)
