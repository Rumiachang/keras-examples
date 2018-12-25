import os
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
#from tensorflow.keras import optimizers
#from tensorflow.keras.utils 
import np_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Input
import tensorflow.keras.callbacks
from tensorflow.keras.optimizers import SGD

import numpy as np
#from smallcnn import save_history

"""
classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']

"""

classes = ['Dog', 'Cat', 'Raccoon', 'Macaque']

#IMAGE_SIZE = 150

BATCH_SIZE = 32
#1バッチの画像数
NUM_TRAINING_SAMPLES = 4000
#トレーニング画像の総枚数
NUM_VALIDATION_SAMPLES = 1000
#テストデータ画像の総枚数
EPOCHS = 50
#エポック数
N_CLASSES = len(classes)
#クラス数
IMG_ROWS, IMG_COLS = 150, 150
#画像の大きさ
CHANNELS = 3
#画像のチャンネル数（RGBなので3）
train_data_dir = 'data/train'
#トレーニングデータのディレクトリ
validation_data_dir = 'data/validation'
#テストデータのディレクトリ

result_dir = 'results'
#リザルトのディレクトリ
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

def train_top_model():
    #InceptionV3のボトルネック特徴量を入力とし、正解クラスを出力とするFCNNを作成する
    input_tensor = Input(shape=(IMG_ROWS, IMG_COLS, CHANNELS))
    #入力テンソル（画像の縦横ピクセルとRGBチャンネルによる3階テンソル）
    base_model = InceptionV3(weights='imagenet', include_top=False,input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #出力テンソルをflatten
    x = Dense(1024, activation='relu')(x)
    #全結合，ノード数1024，活性化関数relu
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       rotation_range=10
                                      )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                        target_size=(IMG_ROWS, IMG_COLS),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        shuffle=True
                                                       )

    validation_generator = test_datagen.flow_from_directory(directory=validation_data_dir,
                                                            target_size=(IMG_ROWS, IMG_COLS),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical',
                                                            shuffle=True
                                                           )
    
    hist = model.fit_generator(generator=train_generator,
                               steps_per_epoch=NUM_TRAINING_SAMPLES//BATCH_SIZE,
                               epochs=EPOCHS,
                               verbose=1,
                               validation_data=validation_generator,
                               validation_steps=NUM_VALIDATION_SAMPLES//BATCH_SIZE,
                              )

    #model.save('vermins_fc_model.hdf5')
    model.save(os.path.join(result_dir, 'vermins_fc_model.h5'))
    save_history(hist, os.path.join(result_dir, 'history_extractor.txt'))
   
    
if __name__ == '__main__':
    #save_bottleneck_features()
    train_top_model()
