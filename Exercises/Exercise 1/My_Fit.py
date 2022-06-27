import matplotlib.pyplot as plt
import numpy as np

figure = plt.figure()
ax = figure.add_subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
C=[] #creating empty array
D=[] #creating empty array

def my_linfit(x , y ) :
    a = 0
    b = 0
    return a , b

def onclick(event):
   
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    plt.plot(event.xdata, event.ydata, ',')
    if event.button == 1:
        C.insert(0,event.xdata)
        D.insert(0,event.ydata)
        figure.canvas.draw()
    else:
        xNew=np.around((np.array(C)), decimals=2)  #creating new numpy array
        yNew=np.around((np.array(D)), decimals=2)  #creating new numpy array
        print(xNew)
        print(yNew)
        a,b = my_linfit(xNew , yNew ) 
        plt.plot(xNew , yNew , 'kx' )
        totalPoints=len(xNew)
        print(totalPoints)
        x = np.arange((min(xNew)-1),(max(xNew)+1) ,0.2)
        a=((totalPoints*(sum(xNew*yNew)))-(sum(xNew)*sum(yNew)))/((totalPoints*(sum(pow(xNew,2))))-(pow(sum(xNew),2))) #equation from the derivation
        b=(sum(yNew)-(sum(xNew)*a))/totalPoints
        ax.plot(x,((a*x)+b),'r')
        figure.canvas.draw()
        print(f"My_fit :_a={b}_and_b={b}")
        
cid = figure.canvas.mpl_connect('button_press_event', onclick)

plt.show()

