import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
#from tensorflow.keras.utils 
import np_utils
import numpy as np
#from smallcnn import save_history

"""
classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']
"""
classes = ['Dog', 'Cat', 'Macaque']
batch_size = 32
nb_classes = len(classes)

img_rows, img_cols = 150, 150
channels = 3

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

#nb_samples_per_class = 120

nb_train_samples = 2000
nb_val_samples = 200
nb_epoch = 50

result_dir = 'results'
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
    
def save_bottleneck_features():
    """VGG16に訓練画像、バリデーション画像を入力し、
    ボトルネック特徴量（FC層の直前の出力）をファイルに保存する"""

    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    model = VGG16(include_top=False, weights='imagenet')
    model.summary()

    # ジェネレータの設定
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # 訓練セットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(os.path.join(result_dir, 'bottleneck_features_train.npy'),
            bottleneck_features_train)

    # バリデーションセットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_validation = model.predict_generator(generator, nb_val_samples)
    np.save(os.path.join(result_dir, 'bottleneck_features_validation.npy'),
            bottleneck_features_validation)


def train_top_model():
    """VGGのボトルネック特徴量を入力とし、正解を出力とするFCネットワークを訓練"""
    # 訓練データをロード
    # ジェネレータではshuffle=Falseなのでクラスは順番に出てくる
    # one-hot vector表現へ変換が必要
    train_data = np.load(os.path.join(result_dir, 'bottleneck_features_train.npy'))
    #train_labels = [i // nb_samples_per_class for i in range(nb_train_samples)]
    #train_labels = np_utils.to_categorical(train_labels, nb_classes)

    # バリデーションデータをロード
    validation_data = np.load(os.path.join(result_dir, 'bottleneck_features_validation.npy'))
    #validation_labels = [i // nb_samples_per_class for i in range(nb_val_samples)]
    #validation_labels = np_utils.to_categorical(validation_labels, nb_classes)

    # FCネットワークを構築
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    history = model.fit(train_data, 
                        epochs=nb_epoch, batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))
    save_history(history, os.path.join(result_dir, 'history_extractor.txt'))


if __name__ == '__main__':
    save_bottleneck_features()
    train_top_model()
