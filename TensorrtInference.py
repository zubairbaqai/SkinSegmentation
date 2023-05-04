
FirstTime=True
import engine as eng
import inference as inf
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import copy

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

serialized_plan_fp32 = "unet.plan"
HEIGHT = 608
WIDTH = 608


def Test(img):
    # import pickle
    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # exit()


    cuda.init()
    device = cuda.Device(0)  # enter your gpu id here
    ctx = device.make_context()

    engine = eng.load_engine(trt_runtime, serialized_plan_fp32)


    h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)

    out = inf.do_inference(engine, img, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)


    print(out.shape)
    out=np.rint(out)

    out=np.reshape(out,(out.shape[2],out.shape[3]))

    out=out*255

    ctx.pop()
    # very important




    return out


# infile = open('object.pickle', 'rb')
# img = pickle.load(infile)
# Test(img)