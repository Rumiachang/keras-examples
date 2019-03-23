import os
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

"""
学習済み重みをロードしてテストデータで精度を求める
"""

result_dir = 'results'

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']

batch_size = 32
nb_classes = len(classes)

img_rows, img_cols = 150, 150
channels = 3
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

model.load_weights(os.path.join(result_dir, 'vermins.h5'))
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

       test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                        target_size=(IMG_ROWS, IMG_COLS),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        shuffle=True
                                                       )

# 学習済みの重みをロード
model.load_weights(os.path.join(result_dir, 'finetuning.h5'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# テスト用データを生成するジェネレータ
test_data_dir = 'test_images'
nb_test_samples = 170
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

# 精度評価
loss, acc = model.evaluate_generator(test_generator, val_samples=nb_test_samples)
print(loss, acc)
