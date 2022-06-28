import matplotlib.pyplot as plt
import numpy as np


def arc(x,y):
    N11=2*(sum(pow(x,2))-1/len(x)*sum(x)*sum(x))
    N12=2*(sum(x*y)-1/len(x)*sum(x)*sum(y))
    N21=N12
    N22=2*(sum(pow(y,2))-1/len(x)*sum(y)*sum(y))
    W1=sum(pow(x,3))+sum(x*pow(y,2))-1/len(x)*sum(pow(x,2))*sum(x)-1/len(x)*sum(pow(y,2))*sum(x)
    W2=sum(pow(x,2)*y)+sum(pow(y,3))-1/len(x)*sum(pow(x,2))*sum(y)-1/len(x)*sum(pow(y,2))*sum(y)
    det=N11*N22-N12*N21
    x0=(W1*N22-W2*N12)/det
    y0=(W2*N11-W1*N21)/det
    R0=np.sqrt(pow(x0,2)+pow(y0,2)+1/len(x)*((sum(pow(x,2))+sum(pow(y,2))-2*(x0*sum(x)+y0*sum(y)))))
    print(x0,' ',y0,' ',R0)
    return x0, y0, R0
def mnk(x,y):
    n=len(x)
    a=(n*sum(x*y)-sum(x)*sum(y))/(n*sum(pow(x,2))-pow(sum(x),2))
    b=(sum(y)-a*sum(x))/n
    #print(a,' ',b)
    return a,b
def liner(x,y,n=15):
    lkj = int(n * np.floor(len(x) / n))
    u=[]
    for i in range(0, lkj, n):
        a, b = mnk(x[i:i + n], y[i:i + n])
        h = np.linspace(x[i + n-1], x[i], n)
        k = a * h + b
        plt.plot(h, k)
        if i > 1:
            u.append(np.arctan(a) * 180 / np.pi - np.arctan(a5) * 180 / np.pi)
            print(np.arctan(a) * 180 / np.pi - np.arctan(a5) * 180 / np.pi)
        a5 = a
    return u
# окружность
R=50
t = np.arange(-np.pi/2, 0, 0.01)          # угол t от 0 до 2pi с шагом 0.01
a0=5
b0=10
x=R*np.sin(t)+a0
y=R*np.cos(t)+b0
g=np.random.normal(0,0.06,len(x))
g1=np.random.normal(0,0.06,len(x))
x=x+g1
y=y+g
#plt.plot(x, y)
#arc(x,y)

# прямая
xl1=[60]*200
yl1=np.linspace(5,70,len(xl1))
g2=np.random.normal(0,0.1,len(xl1))
xl1=xl1+g2
#plt.plot(yl1, xl1)
k1=np.concatenate((x,yl1))
k2=np.concatenate((y,xl1))
#plt.plot(k1,k2)

u=liner(k1,k2,20)
flag=[]
for i in range(len(u)-1):
    if u[i+1]-u[i]>6:
        flag.append(i+1)
print(flag)
for i in flag:
    arc(k1[:i*20], k2[:i*20])
print(mnk(k1[flag[0]*20:], k2[flag[0]*20:]))
plt.axis('equal')
plt.show()

