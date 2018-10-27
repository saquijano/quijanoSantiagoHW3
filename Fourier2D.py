import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft, fft2
from scipy import misc
from skimage import io


arbol=sp.misc.imread("arbol.png",flatten=True)
#arbol2=io.read("arbol.png")/255
print(type(arbol),arbol.size)
plt.imshow(arbol)
plt.show()

#transformada
base,altura=np.shape(arbol)
trans = fft2(arbol)/(base*altura)
SS=1/(altura)
print(np.shape(trans),np.shape(arbol))


freqv=fftfreq(np.shape(trans)[0],SS)
print (freqv)
plt.figure()
plt.xlabel("frecuencia")
plt.plot(freqv,trans)
plt.legend()
plt.show()

graf=np.abs(trans)


#funcion para filtrar transformada
def bajos(freq,sube):
	freqmenor=[]
	lista=[]
	for i in range(len(freq)):
		if (freq[i]<1000 and freq[i]>-1000):
			freqmenor.append(freq[i])
			lista.append(sube[i])
	return freqmenor, lista

#filtr=bajos(freqx,sumar)

#invX=ifft(filtr[1])
