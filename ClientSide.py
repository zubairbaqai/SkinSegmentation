import requests
import numpy as np
import cv2
# image_array =  Input images as a numpy array with dimension [batch_size, height, width, channels]

from tritonclient import http



def run_inference(input_batch):
  """
  Run inference on the input batch of images. Send images to the triton inference
  server and return the response
  :param input_batch: Input batch of images of shape [b, w, h, 3]
  :return: output inference masks of shape [b, 1, w, h]
  """
  triton_client = http.InferenceServerClient(
    url="127.0.0.1:8009", verbose=False)

  input_batch = np.float32(input_batch)
  input_images = http.InferInput("input_1", (input_batch.shape),
                                 'FP32')
  input_images.set_data_from_numpy(input_batch, binary_data=True)
  output = http.InferRequestedOutput("sigmoid", binary_data=True)
  response = triton_client.infer("skin_seg", model_version="1",
                                 inputs=[input_images], outputs=[output])


  masks = response.as_numpy("sigmoid")
  masks[masks > 0.5] = 1
  masks[masks < 0.5] = 0

  mask=np.reshape(masks,(masks.shape[1],masks.shape[2]))

  return mask

