



# # Loading dataset

# For this example we will use **CamVid** dataset. It is a set of:
#  - **train** images + segmentation masks
#  - **validation** images + segmentation masks
#  - **test** images + segmentation masks
#  
# All images have 320 pixels height and 480 pixels width.
# For more inforamtion about dataset visit http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/.

# In[1]:


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


# In[2]:


DATA_DIR = './'
import copy


# In[3]:


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def randomCrop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask
    

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
        # self.ids = os.listdir(images_dir)
        # self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        #
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode=mode
    
    def GetFiles(self, filename,OriginalSizeFlag):
        
        # read data
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(filename)

        print(image.shape)
        print(mask.shape)
        

        
        if(self.mode=="test"):
            widht=image.shape[0]
            height=image.shape[1]
            
            Theshold=512

            if(widht>1200 and height>1200):
                Theshold = 1200
                OriginalSizeFlag=None


            print(OriginalSizeFlag , "CHECK")
            if(OriginalSizeFlag ==None):
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
            
            
                
                
            
                






#             if(widht>1000 or height>1000):
#                 widht=int(widht/4)
#                 height=int(height/4)

#             if(height+widht>1300):
#                 widht=int(widht/2)
#                 height=int(height/2)



            
            height = height - (height % 32)
            widht = widht - (widht % 32)
            

#             height = 320
#             widht = 320



            


            image = cv2.resize(image, (height,widht), interpolation = cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (height,widht), interpolation = cv2.INTER_AREA)

            
        else:

                image = cv2.resize(image, (320,320), interpolation = cv2.INTER_AREA)
                mask = cv2.resize(mask, (320,320), interpolation = cv2.INTER_AREA)
                
                
        
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
    


x_train_dir = os.path.join(DATA_DIR, 'InputFiles')
y_train_dir = os.path.join(DATA_DIR, 'FeetPointsMask')

x_valid_dir = os.path.join(DATA_DIR, 'validImg')
y_valid_dir = os.path.join(DATA_DIR, 'ValidMask')

x_test_dir = os.path.join(DATA_DIR, 'NewTestImages')
y_test_dir = os.path.join(DATA_DIR, 'NewTestImages')


# # Dataloader and utility functions 

# In[ ]:







import albumentations as A
print(A.__version__)


# In[6]:


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


# In[7]:



# # Segmentation model training

# In[1]:


import segmentation_models as sm
sm.set_framework('tf.keras')
# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`


# In[2]:


BACKBONE = 'efficientnetb3'
BATCH_SIZE = 4
CLASSES = ['sky']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)


# In[3]:


# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)




# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]


# In[61]:


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



model.load_weights('best_model.h5') 


test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    classes=CLASSES, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    mode="test"
)



def RunImage(TestImage,filename,OriginalSizeFlag):
    try:
        image, gt_mask,OriginalImage = test_dataset.GetFiles(TestImage,OriginalSizeFlag)
    except Exception as e:
        print(e)
        print("Some problem on the image")
        
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
    

    

    cv2.imwrite("./static/Outputimages/"+ filename, added_image)
    return added_image


    


