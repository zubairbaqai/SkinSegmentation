import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import tensorflow.keras as keras


import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import albumentations as A

import segmentation_models as sm
sm.set_framework('tf.keras')

BACKBONE = 'efficientnetb3'
BATCH_SIZE = 4
CLASSES = ['sky']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

optim = tf.keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)




model.load_weights('best_model.h5')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def RunImage(TestImage,filename):





    image = cv2.cvtColor(TestImage, cv2.COLOR_BGR2RGB)
    print(image.shape)
    mask=copy.deepcopy(image)

    Preprocess=get_preprocessing(preprocess_input)


    widht = image.shape[0]
    height = image.shape[1]

    Theshold = 608

    if (height < 300 or widht < 300):
        pass
    else:
        if (widht > height):
            Ratio = height / Theshold
            height = Theshold
            widht = widht / Ratio


        else:
            Ratio = widht / Theshold
            widht = Theshold
            height = height / Ratio

    height = int(height - (height % 32))
    widht = int(widht - (widht % 32))



    image = cv2.resize(image, (height, widht), interpolation=cv2.INTER_CUBIC)
    OriginalImage = copy.deepcopy(image)
    mask = cv2.resize(mask, (height, widht), interpolation=cv2.INTER_CUBIC)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # extract certain classes from mask (e.g. cars)
    masks = [(mask == v) for v in [0]]
    mask = np.stack(masks, axis=-1).astype('float')

    # add background if mask is not binary
    if mask.shape[-1] != 1:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)

    sample = Preprocess(image=image, mask=mask)

    image, mask = sample['image'], sample['mask']

    image = np.expand_dims(image, axis=0)



    pr_mask = model.predict(image).round()
    pr_mask = np.squeeze(pr_mask, axis=0)
    pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2BGRA)

    pr_mask[np.all(pr_mask == (0, 0, 0, 1), axis=-1)] = (0, 255, 0, 255)
    pr_mask[np.all(pr_mask == (1, 1, 1, 1), axis=-1)] = (0, 0, 0, 0)

    OriginalImage = cv2.cvtColor(OriginalImage, cv2.COLOR_RGB2RGBA)

    print(OriginalImage.shape)
    print(pr_mask.shape)

    OriginalImage = np.asarray(OriginalImage, np.float64)
    pr_mask = np.asarray(pr_mask, np.float64)

    added_image = cv2.addWeighted(OriginalImage, 1, pr_mask, 0.5, 0)
    # print(np.unique(pr_mask))

    #print(test_dataset.images_fps[i].split("/")[-1])

    cv2.imwrite("./static/Outputimages/"+ filename, added_image)

    return added_image
