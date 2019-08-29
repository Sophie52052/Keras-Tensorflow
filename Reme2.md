# 主題：Deep Neural network with Keras 講者：黃睦翔
###### tags: `Keras` `MachineLearning`  `CoreLayer` `Dense` `RNN` `LSTM` `GRU` `Activation` `Depthwise` `Pointwise` `Sampling` `Pooling` `Padding` `Stride`

* Core Layer
![](https://i.imgur.com/vl9OPxB.png)
https://keras-cn.readthedocs.io/en/latest/layers/core_layer/

* Dense：
1.全連接層
2.實現：output = activation(dot(input, kernel) + bias)
**activation**：按逐元素計算的激活函數
**kernel**：由[網絡層](https://keras.io/zh/layers/about-keras-layers/)(有很多共同函數)創建的權值矩陣
![](https://i.imgur.com/WhvBKcL.png)
**bias**：其偏置向量(use_bias為True時才有用)
3.Keras語法：keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
**units** :輸出的空間維度
**activation** :沒寫則不使用激活函數(即線性: a(x) = x)
**use_bias** :偏置向量
**kernel_initializer** : kernel權值矩陣的[初始化器](https://keras.io/zh/initializers/)
EX.Zeros、Ones、Constant(常數初始化器)、RandomNormal(有平均/標準差/seed)...等
**bias_initializer** :偏置向量的初始化器(與上相同)
**kernel_regularizer** :kernel權值矩陣的[正則化函數](https://keras.io/zh/regularizers/)
![](https://i.imgur.com/DZZrKIK.png)
**bias_regularizer** :偏置向的的正則化函數(同上)
**activity_regularizer** :dense輸出的正則化函數(它的"activation")(同上)
**kernel_constraint** :kernel權值矩陣的[約束函數](https://keras.io/zh/constraints/)
EX.MaxNorm、NonNeg、UnitNorm、MinMaxNorm
**bias_constraint** :偏置向量的約束函數(同上)。
**EX.** model.add(Dense(32))

* Activation
![](https://i.imgur.com/BhH7IKS.png)
![](https://i.imgur.com/GpQTPRz.png)
![](https://i.imgur.com/8RmRwT7.png)
![](https://i.imgur.com/R1E9Xr0.png)
1.Sigmoid：0~1之間
2.hyperbolic tangemt funtion -1 1
3.rectified linear funtion
4.leakyrectified linear funtion
5.1.Softmax：每一個元素的範圍都在0~1之間，並且所有元素的和為1

* Dropout
1.背景：機器學習容易過擬合、費時
PS.過擬：訓練Loss小，準確率高；但在測試上Loss大，準確率低
2.作法：
*隨機（臨時）刪掉一半的隱藏神經元，輸入輸出保持不變
![](https://i.imgur.com/Q4wG73u.png)
*把輸入通過修改前向傳播，把損失結果反向傳播
*執行完在沒被刪除的神經元上，按隨機梯度下降法更新對應參數（w，b）
*Keras語法：重複這過程：
恢復被刪掉的神經元（被刪除者保持原樣，沒被刪除者更新）
從隱藏層中隨機選擇一半大小子集刪除（備份被刪除神經元的參數）
對一小批訓練樣本，先前向傳播然後反向傳播損失並根據隨機梯度下降法更新參數（w，b） （沒有被刪除的那一部分參數得到更新，刪除的神經元參數保持被刪除前的結果）
3.keras.layers.core.Dropout(rate, noise_shape=None, seed=None)

* Flatten
1.作法：把輸入一維化，常用在從卷積層到全連接層的過渡
PS.Flatten不影響batch的大小(批)。
2.Keras語法：keras.layers.core.Flatten()
**EX.** model.add(Flatten())

* Input
1.實例化Keras 張量
2.Keras語法：keras.engine.input_layer.Input()
**EX.** x = Input(shape=(32,))

* Permute
1.置換輸入的維度
2.Keras語法：keras.layers.Permute(dims)
**EX.** model.add(Permute((2, 1), input_shape=(10, 64)))

* Sequential Model
1.Easy but not so flexible(Sequential-一條路走到底)
2.作法：
向Sequential模型傳遞一個層的列表來構造該模型，可用add，須知輸入數據size
![](https://i.imgur.com/knhJBlR.png)
![](https://i.imgur.com/sHdw3Mm.png)
![](https://i.imgur.com/NyM7No6.png)
訓練模型前，需要compile配置：
[optimizer優化器](https://keras-cn.readthedocs.io/en/latest/other/optimizers/)：compile必要參數，優化器內裡面參數可調
[loss損失函數](https://keras-cn.readthedocs.io/en/latest/other/objectives/)：compile必要參數，試圖最小化loss
metrics指標列表：一般設為accuracy，名子可為預定或用戶定制的函數指標，函數應該返回單個張量或字典
![](https://i.imgur.com/Yf6Ntss.png)
3.例子：https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
4.VS.Functional Model--較佳

* Simple RNN
![](https://i.imgur.com/Bob23y1.png)
![](https://i.imgur.com/oavXvAo.png)
輸入層到隱含層權重由U表示，它將原始輸入進行抽象作為隱含層的輸入。
隱含層到隱含層的權重為W，它是網路的記憶控制者，負責調度記憶。
隱含層到輸出層的權重為V，從隱含層學習到的表示將通過它再一次抽象，並作為最終輸出。
1.**前向傳播(Forward Propagation)**
t=0時--UVW隨機初始化，h0通常為0
![](https://i.imgur.com/cnK2lyb.png)
f(.)是隱含層的激活函數tanh、relu，sigmoid等，g(.)是輸出層的激活函數Softmax，h1為第一層的輸出，y1為預測的結果。
![](https://i.imgur.com/3Z02G6u.png)
誤差：ei-第i個時間步的誤差，y-預測結果，d-實際結果，誤差函數fe()可以選擇是交叉商(Cross Entropy)
![](https://i.imgur.com/HUFvegA.png)
2.**反向傳播(Back Propagation)**
利用輸出層誤差e，求權重梯度，▽V，▽U，▽W，利用梯度下降法更新個個權重
![](https://i.imgur.com/gcatdFp.png)
各權重梯度
![](https://i.imgur.com/3vn61iv.png)
▽V，由於它不依賴之前的狀態，可以直接求導獲得
![](https://i.imgur.com/y0x5WS3.png)
▽U，▽W依賴之前的狀態，不能直接求導，需定義中間變量
![](https://i.imgur.com/xbaA0Ig.png)
先計算出輸出層的δy，在向後傳播至各層δh，依次類推，直至輸入層
![](https://i.imgur.com/GEmTcoa.png)
*表示點積，只要計算δy及所有δh，即可算出▽U，▽W
![](https://i.imgur.com/K9AMFbM.png)
3.[Keras語法](https://keras.io/zh/layers/recurrent/)：
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
> http://arbu00.blogspot.com/2017/05/3-rnn-recurrent-neural-networks.html

* LSTM & GRU
![](https://i.imgur.com/ijdz2AC.png)
![](https://i.imgur.com/ClvYEiR.png)
![](https://i.imgur.com/Z0ohRdn.png)
**LSTM**
[keras語法](https://keras.io/zh/layers/recurrent/)：keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
**GRU**
[keras語法](https://keras.io/zh/layers/recurrent/)：keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)

* CNN-1D
1.1D一維卷積層（即時域卷積）：一維輸入信號上進行鄰域濾波
2.使用：使用該層作為首層時，需有input_shape。EX.(10,128)代表一個長為10的序列，序列中每個信號為128向量。而(None, 128)代表變長的128維向量序列。
3.input shape（samples，steps，input_dim）的3D张量
output shape（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，steps的值会改变
4.keras語法：keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

* CNN-2D
1.二維卷積層(即空域卷積)：該層對二維輸入進行滑動窗卷積
2.使用：當使用該層作為第一層時，提供應input_shape參數。EX.input_shape = (128,128,3)代表128×128的彩色RGB圖像
3.**channels_first**--input（samples,channels，rows，cols）的4D张量，output（samples，nb_filter, new_rows, new_cols）的4D张量
**channels_last**--input（samples，rows，cols，channels）的4D张量，output（samples，new_rows, new_cols，nb_filter）的4D张量
4.keras語法：keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
![](https://i.imgur.com/PFuTMZy.png)
PS.
padding：vaild和same(中間算很多次 周圍少)EX補0
Stride：跨幾步
Batch：我們在 training 的時候，不會只挑一個 data 出來，而是一次挑一把 data 出來 train
假設我們一次挑出 3 筆 data 出來 train，我們的 Batch = 3
P自訂、S表示前進數 nXn * fXf — ((N+2P-F)/S +1)^2
* Q&A
Q：(batch, 128, 128, 32)   Conv2D    (batch, 128, 128, 128)
How many filters we need?
A:n*n*128


* Depthwise convolution
1.**input**: 寬 * 高 * channel數
**Kernel Map**:寬 * 高 * kernel數(kernel寬高假設一樣)
**output**:寬 * 高 * channel數
![](https://i.imgur.com/dCyOd73.png)
2.輸入資料每個Channel都建立一個k * k的Kernel，然後每一個Channel針對對應的Kernel都各自(分開)做convolution
![](https://i.imgur.com/jONR46D.png)

* Pointwise convolution
1.**input**: 寬 * 高 * Nch
**Kernel Map**:Nk個 (1 * 1)
**output**:寬 * 高 * Nk
![](https://i.imgur.com/IoUYmGu.png)
![](https://i.imgur.com/T524qYg.png)
![](https://i.imgur.com/uwFCvPY.png)
2.建立Nk個1*1*Nch的kernel Map，將depthwise convolution的輸出做一般1 * 1的卷積計算

* Pooling
1.池化對圖片中的每個小塊（而不是每個點）提取信息
2.常用方法：有最大值法和均值法
![](https://i.imgur.com/B3pVSm6.png)

*  Sampling
1.上採樣原理：圖像放大幾乎都是採內插值，在在像素點之間採用合適的插值算法插入新的元素。
2.下採樣原理：對於一幅圖像I尺寸為M*N，對其進行s倍下採樣，即得到(M/s)*(N/s)尺寸的得分辨率圖像，當然s應該是M和N的公約數才行，如果考慮的是矩陣形式的圖像，就是把原始圖像s*s窗口內的圖像變成一個像素，這個像素點的值就是窗口內所有像素的均值


> https://docs.google.com/a/iir.csie.ncku.edu.tw/viewer?a=v&pid=sites&srcid=aWlyLmNzaWUubmNrdS5lZHUudHd8aWlyLWxhYnxneDo2OTFkMjNjMDNhODMzMzM1

