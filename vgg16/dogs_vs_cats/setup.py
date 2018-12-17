import os
import glob
"""
train.zipを解凍したtrainから
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
にあるように訓練データを振り分ける
"""

imgs_path = "./image_dir/*.png"
source_dir = "./train"
train_dir = "./data/train"
valid_dir = "./data/validation"

os.makedirs("%s/dogs" % train_dir)
os.makedirs("%s/cats" % train_dir)
os.makedirs("%s/dogs" % valid_dir)
os.makedirs("%s/cats" % valid_dir)

"""
list_imgs_path = glob.glob(imgs_path)
for i, imgs_file in enumerate(list_imgs_path):
    os.rename(imgs_file ,"./image_dir/" + str(i) + ".png" )
"""


# 最初の1000枚の画像をtrain_dirに移動
for i in range(1000):
    os.rename("%s/dog.%d.jpg" % (source_dir, i + 1),
              "%s/dogs/dog%04d.jpg" % (train_dir, i + 1))
    os.rename("%s/cat.%d.jpg" % (source_dir, i + 1),
              "%s/cats/cat%04d.jpg" % (train_dir, i + 1))

# 次の400枚の画像をvalid_dirに移動
for i in range(400):
    os.rename("%s/dog.%d.jpg" % (source_dir, 1000 + i + 1),
              "%s/dogs/dog%04d.jpg" % (valid_dir, i + 1))
    os.rename("%s/cat.%d.jpg" % (source_dir, 1000 + i + 1),
              "%s/cats/cat%04d.jpg" % (valid_dir, i + 1))

