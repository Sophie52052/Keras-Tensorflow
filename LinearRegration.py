

import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
@contextmanager

def myplot(fig, subplot, size=6):
    #設定背景
    ax = fig.add_subplot(subplot)
    ax.set_xlim(0, 6) #X軸
    ax.set_ylim(0, 6) #Y軸
    ax.grid(axis='y') #格線     
    yield ax # Plot the data
   
fig = plt.figure(figsize=(5, 5)) #圖的大小 

with myplot(fig, 111) as ax:
    #畫線、加入雜訊
    x_p=np.arange(0,7,1)
    b=np.random.rand(7)
    b=b*1.03+0.05
    a=np.sort(np.random.rand(7))
    a=a*1.01+0.1
    y_p=a*x_p+b
    ax.plot(x_p,y_p,"o")    
    #a=0.9,b=0.2
    x=np.arange(0,7,0.5)
    y=0.9*x+0.2
    ax.plot(x,y,"o")    
    
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

    
print('ap='+str(ap),'bp='+str(bp))

 