'''import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5], dtype=float)
y=np.array([2,4,6,8,10],dtype=float)
w,b=0.0,0.0
alpha=0.01
epochs=1000
n=len(x)
for _ in range(epochs):
    y_pred=w* x+b
    loss=(1/n)>np.sum((y-y_pred)**2)
    dw=-(2/n)*np.sum(x+(y-y_pred))
    db=-(2/n)*np.sum(y-y_pred)
    w-=alpha*dw
    b-=alpha*db
    if _ % 100==0:
        print(f"Epoch{_},loss:{loss:.4f}")
print(f"leraned parameters :w={w:.4f},b={b:.4f}")
plt.scatter(x,y)
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
# generating random data
n=1000
m=3
x=np.random.randn(n,m)*10
true_w=np.array([2.0,-1.5,0.5])#features
true_b=3.0 #no. of features in bias
y=np.dot(x,true_w)+true_b+np.random.randn(n)*2
#y=w1*w1+w2*w2+w3*w3+b+noise
w=np.zeros(m)
b=0.0
alpha=0.01
epochs=1000
for epoch in range(epochs):
    y_pred=np.dot(x,w)+b
    loss=(1/n)*np.sum(y-y_pred**2)
#gradients
    dw=-(2/n)*np.dot(x.T,(y-y_pred))
    db=-(2/n)*np.sum(y-y_pred)
    w-=alpha*dw
    b-=alpha*db
    if epoch % 100==0:
        print(f"Epoch {epoch},loss:{loss:.4f}")
print(f"weights original:{true_w},true bias={true_b}")
print(f"learned weights:{w}, learned bias:{b:.4f}")
plt.scatter(x[:,0],y,color='blue',alpha=0.5,label='Data')
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
np.random.seed(42)
x=np.sort(5*np.random.rand(80,1),axis=0)
y=np.sin(x).ravel()+np.random.normal(0,0.1,x.shape[0])
x=x.reshape(-1,1)
degrees=[1,2,5,10]
plt.figure(figsize=(10,6))
for degree in degrees:
    preg=make_pipeline(PolynomialFeatures(degree),LinearRegression)
    preg.fit(x,y)
    x_smooth=np.linspace(x.min(),x.max(),300).reshape(-1,1)
    y_smooth=preg.predict(x_smooth)
    plt.plot(x_smooth,y_smooth)
plt.legend()
plt.grid(True)
plt.show()'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(42)
n_samples=100
x1=np.random.uniform(-3,3,n_samples)
x2=np.random.uniform(-3,3,n_samples)
x=np.column_stack((x1,x2))
y=np.sin(x1)*np.cos(x2)+0.1*np.random.randn(n_samples)
degrees=[1,2,3]
x1_min,x1_max=x1.min() -1,x1.max()+1
x2_min,x2_max=x2.min() -1,x2.max()+1
xx1,xx2=np.meshgrid(np.linspace(x1_min,x1_max,50),np.linspace(x2_min,x2_max,50))
x_grid=np.c_[xx1.ravel(),xx2.ravel()]
fig=plt.figure(figsize=(15,5))
for i,degree in enumerate(degrees,1):
    polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(x,y)
    y_pred=polyreg.predict(x_grid).reshape(xx1.shape)
    ax=fig.add_subplot(1,len(degrees),i,projection='3d')
    ax.scatter(x1,x2,v,color='black',s=20,label='Data Points')
    ax.plot_surface(xx1,xx2,y_pred,cmap='virdis',alpha=0.0)
    ax.legend()
plt.tight_layout()
plt.show()