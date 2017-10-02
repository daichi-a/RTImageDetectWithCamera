# -*- coding: utf-8 -*-
# 参考: http://qiita.com/juntaki/items/9a13a3d2217ca223cf03
from LoadImageDataForKeras import LoadImageDataForKeras
import numpy as np
from PIL import Image

#import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
import glob



# 順番に読み込んでも，トレーニングデータとテストデータをミックスして出力してくれる関数
# scikit-learnというパッケージに入っている
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    # イメージを読み込むためのクラスのインスタンス化
    loadingObj = LoadImageDataForKeras(2)
    # 2は何分類問題かを示す．ここでは2分類問題

    # get_image_array_for_kerasには，ファイル名とラベルを与える
    # 2分類問題のラベルが0の時はnp.array([[1, 0]], dtype=np.uint8)
    # ラベル「1」の時はnp.array([[0, 1]], dtype=np.uint8)を与える

    # 3分類問題であれば，
    # np.array([[1, 0, 0]]), np.array([[0, 1, 0])), np.array([[0, 0, 1]])
    # のどれかになる

    # りんご画像ringo0~ringo9.jpgを読み込む
    for i in range(10):
        loadingObj.get_image_array_for_keras('ringo'+str(i)+'.jpg', np.array([[1, 0]], dtype=np.uint8))

    # みかん画像mikan0~mikan9.jpgを読み込む
    for i in range(10):
        loadingObj.get_image_array_for_keras('mikan'+str(i)+'.jpg', np.array([[0, 1]], dtype=np.uint8))

    # この出力が(サンプル数, 3, 50, 50)になってれば良い
    print(loadingObj.get_stacking_image_array().shape) 
    # 積み重ねた全体を出力する
    # print(loadingObj.get_stacking_image_array())

    #この出力が(サンプル数, 分類する数)になってれば良い
    print(loadingObj.get_stacking_label_array().shape) 
    # 積み重ねた全体を出力する
    # print(loadingObj.get_stacking_label_array())

    # 読み込んだデータを，ミックスしてトレーニングデータとテストデータに分ける
    # 参照: http://stackoverflow.com/questions/3674409/numpy-how-to-split-partition-a-dataset-array-into-training-and-test-datasets
    data_train, data_test, labels_train, labels_test = \
        train_test_split(loadingObj.get_stacking_image_array(),
                         loadingObj.get_stacking_label_array(),
                         test_size=0.10, # 9割をトレーニングデータ，1割をテストデータ
                         random_state=10)
    print(data_train.shape)
    print(data_test.shape)
    print(labels_train.shape)
    print(labels_test.shape)
    

    # ニューラルネットの定義
    # コンボリューション層とMaxpoolingを適当に重ねたもの
    # MaxPlooling2Dのオプションに関しては以下を参照
    # https://stackoverflow.com/questions/39815518/keras-maxpooling2d-layer-gives-valueerror
    model = Sequential()
    model.add(Convolution2D(96, 3, 3, border_mode="same", activation="relu" ,input_shape=(3, 50, 50) ))
    model.add(Convolution2D(96, 3, 3, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    model.add(Convolution2D(96, 3, 3, border_mode="same", activation="relu"))
    model.add(Convolution2D(96, 3, 3, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    model.add(Convolution2D(96, 3, 3, border_mode="same", activation="relu"))
    model.add(Convolution2D(96, 3, 3, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("sigmoid"))
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adadelta", metrics=['accuracy'])

    # 学習の実行
    # 20エポック回す
    hist = model.fit(data_train, labels_train, nb_epoch=20, batch_size=32, validation_data=(data_test, labels_test))

    # 結果をmatplotlibでplot
    # http://qiita.com/TypeNULL/items/4e4d7de11ab4361d6085
    #loss = hist.history['loss']
    #val_loss = hist.history['val_loss']

    #nb_epoch = len(loss)
    #plt.plot(range(nb_epoch), loss, marker='.', label='loss')
    #plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
    #plt.legend(loc='best', fontsize=10)
    #plt.grid()
    #plt.xlabel('epoch')
    #plt.ylabel('loss')
    #plt.show()


    # 学習したネットワークを保存し，再度読み込み，使えることを確認する
    # 参照: http://m0t0k1ch1st0ry.com/blog/2016/07/17/keras/

    # 学習したネットワークをJSONで保存
    model_json_str = model.to_json()
    # 'ringo_or_mikan_model.json'というファイルにネットワーク(モデル)を保存
    open('ringo_or_mikan_model.json', 'w').write(model_json_str)
    # 'ringo_or_mikan_weights.hdf5'というファイル名のhdf5形式で学習結果(重み)を保存
    model.save_weights('ringo_or_mikan_weights.hdf5');

    # ただしhdf5で保存したものをkeras-jsで読むためには，
    # json形式にエンコードする必要があるらしい
    # それに必要なプログラムはkeras-jsに含まれている
    # http://nonbiri-tereka.hatenablog.com/entry/2016/10/17/073541

    # 再度読み込み
    # Pythonはhdf5形式もそのまま読める
    # まずネットワークを保存したファイルを開いてネットワークを読み込み
    learned_model = model_from_json(open('ringo_or_mikan.json').read())
    # そのネットワークに学習した重みを読み込む
    learned_model.load_weights('ringo_or_mikan_weights.hdf5')
    
    learned_model.summary()
    # 使えるようにコンパイルする
    learned_model.compile(loss='binary_crossentropy', optimizer="adadelta", metrics=['accuracy'])
    
    # ネットワークの評価をする
    score  = learned_model.evaluate(data_test, labels_test, verbose=0)
    print('Test loss : ', score[0])
    print('Test accuracy : ', score[1])

    # ネットワークを使う
    # 学習に使用しなかったリンゴ画像を読み込む
    ringo_x = np.array(Image.open('ringo_x.jpg').resize((50, 50)))
    ringo_x = ringo_x.transpose(2, 0, 1)
    stacking_image_array = np.empty((0, 3, 50, 50), dtype=np.uint8)
    stacking_image_array = np.append(stacking_image_array,\
                                     np.array([ringo_x]),\
                                     axis=0)
    print(stacking_image_array.shape)
    # ネットワークによる分類推測を行う
    result = learned_model.predict(stacking_image_array, 
                                         batch_size=1, verbose=1)
    print(result)
    # どのクラスに属するのかの確率を配列で出してくれる
    # predict_class関数を使うと，一番確率が高いクラスをそのまま出力してくれる

    # 学習に使用しなかったミカン画像を読み込み，同じように
    mikan_x = np.array(Image.open('mikan_x.jpg').resize((50, 50)))
    mikan_x = mikan_x.transpose(2, 0, 1)
    stacking_image_array = np.empty((0, 3, 50, 50), dtype=np.uint8)
    stacking_image_array = np.append(stacking_image_array,\
                                     np.array([mikan_x]),\
                                     axis=0)
    print(stacking_image_array.shape)
    result = learned_model.predict_proba(stacking_image_array, 
                                         batch_size=1, verbose=1)
    # 2分類問題だと，predict関数とpredict_proba関数の出力は同じ？
    print(result)
    
