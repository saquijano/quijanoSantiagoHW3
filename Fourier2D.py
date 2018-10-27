import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft, fft2
from scipy import misc
from skimage import io


arbol=sp.misc.imread("arbol.png",flatten=True)
plt.imshow(arbol)
plt.show()

#transformada
base,altura=np.shape(arbol)
trans = fft2(arbol)/(base*altura)
SS=1/(altura*base)
print(np.shape(trans),np.shape(arbol))


freqv=fftfreq(np.shape(trans)[0],SS)
print (freqv)
plt.figure()
plt.xlabel("frecuencia")
plt.plot(freqv,trans)
plt.legend()
#plt.show()

graf=np.abs(trans)


#funcion para filtrar transformada
def bajos(freq,sube):
	freqmenor=np.zeros(256)
	lista=[]
	for i in range(np.shape(freq)[0]):
		for j in range(np.shape(freq)[1]):
			if (freq[i,j]<1000 and freq[i,j]>-1000):
				freqmenor[i,j]=(freq[i])
	return freqmenor, lista

def bajosen(freq,sube):
	freqmenor=np.linspace(0,0,len(freq))
	lista=np.linspace(0,0,len(sube))
	for i in range(len(freq)):
		if (freq[i]<2000 and freq[i]>-2000):
			freqmenor[i]=freq[i]
			#lista[i]=sube[i]
	return freqmenor#, lista

filtr=bajosen(freqv,trans)
plt.figure()
plt.plot(filtr,trans)
plt.show()

#invX=ifft(filtr[1])
