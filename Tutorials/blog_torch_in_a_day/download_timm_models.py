import timm

MODEL_LIST = timm.list_models()

MODEL_LIST = [
    'resnet34'
]

for model in MODEL_LIST:
    pretrained_resnet_34 = timm.create_model(model, pretrained=True)
    
    
# from fastai.vision.all import *

# path = untar_data(URLs.PETS) / 'data'
# dls = ImageDataLoaders.from_name_func(
#     path, get_image_files(path), valid_pct=0.2,
#     label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
    
# # if a string is passed into the model argument, it will now use timm (if it is installed)
# learn = vision_learner(dls, 'vit_tiny_patch16_224', metrics=error_rate)

# learn.fine_tune(1)