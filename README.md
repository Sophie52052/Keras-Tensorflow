# 主題：Machine Learning HW1 講者：柏勳
###### tags:`MachineLearning` `LinearRegression` `HW`

* Q1
實作linear regression : y = ax + b 
理解 loss function
實作 gradient decent
不能使用sklearn等套件
![](https://i.imgur.com/91PJGnI.png)
![](https://i.imgur.com/quWB8m5.png)
![](https://i.imgur.com/fjVe6Vc.png)

* A1
![](https://i.imgur.com/zbBAZTC.jpg)
![](https://i.imgur.com/N4D6hu4.jpg)
![](https://i.imgur.com/BgjSygY.jpg)
```
#開始預測
ap=0.2;#自訂開始a與b
bp=0.2;
   
for i in range(5000) : #epochs
    
    yp=x_p*ap+bp #預測線
        
    loss_mse = np.mean((y_p - yp)**2) #loss function用Y
    
    Da=(-2/7)*sum(y_p-yp)*x_p #對ab做篇微分
    Db=(-2/7)*sum(y_p-yp) #7--N個數
    
    ap=ap-0.002*Da #0.002--LearningRate
    bp=bp-0.002*Db
```