# -*- coding: utf-8 -*-
# Keras用の画像データ作成
# 参考: http://qiita.com/juntaki/items/9a13a3d2217ca223cf03

import numpy as np
from PIL import Image

class LoadImageDataForKeras():
    def __init__(self, label_dimension):
        # label_dimenstion: 分類する数

        # Kerasに読み込ませる配列を作る
        # [サンプル数, 3, 50, 50]の4次元配列になっていなければならない
        # まず最初に空の配列を作る データ型はuint8(unsigned char)つまり0~255の範囲
        self.stacking_image_array = np.empty((0, 3, 50, 50), dtype=np.uint8)

        # ラベル(ここでは画像の2分類問題なので[サンプル数, 2]の2次元配列となる
        # つまり，画像がラベル「0」で表される時は[1, 0]
        # 画像がラベル「1」で表される時は[0, 1]
        # を積み重ねていく
        self.stacking_label_array = \
            np.empty((0, label_dimension), dtype=np.uint8) 

    def get_image_array_for_keras(self, file_name, label_array):
        # file_name: ファイル名
        # label_array: numpy配列で[1, 0] or [0, 1]みたいに
        # どの属性に
        
        im_reading = np.array(Image.open(file_name).resize((50, 50)))
        
        # 折り畳み2Dフィルタを適用するために，
        # インターリーブから，R50*50, G50*50, B50*50という形になるように，
        # 順番を入れ替える (3次元配列を入れ返す[50, 50, 3]から，[3, 50, 50]へ変換
        # 引数は[0,1,2]を[2,0,1]に並べ直すという意味
        im_reading = im_reading.transpose(2, 0, 1)
        
        # 読み込んで末尾に追加する
        # 3次元を積み重ねて4次元配列にするので，vstack, dstackは使えない
        # しょうがないので，遅いのはわかっているがappendする
        self.stacking_image_array = \
            np.append(self.stacking_image_array, np.array([im_reading]), axis=0)
    
        # ラベルを重ねる (ここではこの画像はラベル「0」とする)
        self.stacking_label_array = \
            np.append(self.stacking_label_array, label_array, axis=0)


        # 一番最初はZeroで初期化しているので，そこを取り除く．
        # images = np.delete(images)
        # labels = np.delete(label)

    def get_stacking_image_array(self):
        return self.stacking_image_array

    def get_stacking_label_array(self):
        return self.stacking_label_array
