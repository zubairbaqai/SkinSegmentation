

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


# In[3]:


DATA_DIR = './'
import copy




# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            mode="Other"
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode=mode
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        maskPath=self.masks_fps[i]

        mask = cv2.imread(maskPath)
        

        
        if(self.mode=="test"):
            widht=image.shape[0]
            height=image.shape[1]
            
            Theshold=512
            
            
            if(height<300 or widht<300):
                pass
            else:
                if(widht>height):
                    Ratio=height/Theshold
                    height=Theshold
                    widht=widht/Ratio


                else:
                    Ratio=widht/Theshold
                    widht=Theshold
                    height=height/Ratio
                
            
            height=int(height)
            widht=int(widht)
            

            height = height - (height % 32)
            widht = widht - (widht % 32)

            


            image = cv2.resize(image, (height,widht), interpolation = cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (height,widht), interpolation = cv2.INTER_AREA)

            

        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


        
        ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        
        #mask=mask.reshape(mask.shape[0],mask.shape[1],1)



        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        

        
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
            
            
        
        
        PreviousSize=image.shape
        MaskeShape = mask.shape
        # apply augmentations
        if self.augmentation and not self.mode=="test":
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        #image,mask=randomCrop(image,mask,320,320)
        #image = cv2.resize(image, (320,320), interpolation = cv2.INTER_AREA)
        #mask = cv2.resize(mask, (320,320), interpolation = cv2.INTER_AREA)


        originalImage=copy.deepcopy(image)
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
    
            
        if(self.mode=="test"):
            return image, mask , originalImage
        else:
            return image, mask 
            
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   


x_test_dir = "static/uploads/"




import albumentations as A
print(A.__version__)


# In[38]:


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        #A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        #A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                #A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.3,
        ),

        A.OneOf(
            [
                #A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.3,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                #A.HueSaturationValue(p=1),
            ],
            p=0.3,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

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



import segmentation_models as sm
sm.set_framework('tf.keras')
# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`


# In[10]:


BACKBONE = 'efficientnetb3'
BATCH_SIZE = 4
CLASSES = ['sky']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)


# In[11]:


# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)



# In[13]:


# define optomizer
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


# In[48]:


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices()) # list of DeviceAttributes

model.load_weights('best_model.h5') 



def Run():


    test_dataset = Dataset(
        x_test_dir,
        x_test_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
        mode="test"
    )

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)




    ids = range(len(test_dataset))

    print(len(ids))


    for i in ids:
        try:
            image, gt_mask,OriginalImage = test_dataset[i]
        except Exception as e:
            print(e)
            print("Some problem on the image")
            continue
        image = np.expand_dims(image, axis=0)


        pr_mask = model.predict(image).round()





        pr_mask=np.squeeze(pr_mask,axis=0)
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2BGRA)



        pr_mask[np.all(pr_mask == (0, 0, 0,1), axis=-1)] = (0,255,0,255)
        pr_mask[np.all(pr_mask == (1, 1, 1,1), axis=-1)] = (0,0,0,0)

        OriginalImage = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2RGBA)

        OriginalImage=np.asarray(OriginalImage, np.float64)
        pr_mask=np.asarray(pr_mask, np.float64)

        added_image = cv2.addWeighted(OriginalImage,1,pr_mask,0.5,0)
        #print(np.unique(pr_mask))



        print(test_dataset.images_fps[i].split("/")[-1])

        cv2.imwrite("Outputimages"+"/"+str(test_dataset.images_fps[i].split("/")[-1])+".png",added_image)























