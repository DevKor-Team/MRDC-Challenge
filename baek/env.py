
IMG_SIZE = 384
TRAIN_IMAGE_FOLDER = "/ssd/MRDC/Images/train/rgb"
TEST_IMAGE_FOLDER = "/ssd/MRDC/Images/test"
BATCH_SIZE = 32
# 'vit_base_patch16_384', # 'mobilenetv3_rw'
MODEL_ARCH = "convnext-base-384"
CLASSES = 3
LR = 1e-4
EARLY_STOPPING = True
EPOCHS = 50
WEIGHT_DECAY = 1e-6
