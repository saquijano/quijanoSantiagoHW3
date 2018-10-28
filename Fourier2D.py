import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2
from scipy import misc
from skimage import io


arbol=sp.misc.imread("arbol.png",flatten=True)
plt.imshow(arbol)


#transformada
base,altura=np.shape(arbol)
trans = fft2(arbol)/(base*altura)
SS=1/(altura*base)




freqv=fftfreq(np.shape(trans)[0],SS)
freqh=fftfreq(np.shape(trans)[1],SS)
print(freqv,freqh)
plt.figure()
plt.title("Transformada de fourier")
plt.xlabel("frecuencia")
plt.plot(freqv,trans)
plt.legend()

graf=np.abs(trans)


#funcion para filtrar transformada
def bajos(freq,sube):
	freqmenor=np.zeros((256,256))
	lista=np.zeros((256,256))
	for i in range(np.shape(sube)[0]):
		for j in range(np.shape(sube)[1]):
			if (freq[i,j]<1000 and freq[i,j]>-1000):
				freqmenor[i,j]=freq[i,j]
				lista[i,j]=sube[i,j]
	return freqmenor, lista
frecuencia=np.zeros((256,256))
for i in range(np.shape(frecuencia)[0]):
	frecuencia[:,i]=freqv

def bajosen(freq,sube):
	freqmenor=np.linspace(0,0,len(freq))
	lista=np.linspace(0,0,len(sube))
	for i in range(len(freq)):
		if (freq[i]<2000 and freq[i]>-2000):
			freqmenor[i]=freq[i]
			#lista[i]=sube[i]
	return freqmenor#, lista


filtr=bajos(frecuencia,trans)
print(trans)

plt.figure()
plt.title("Transformada filtrada")
plt.plot(filtr[0],filtr[1])
plt.ylabel("frecuencia")
plt.xlabel(" ")
plt.savefig("quijanoSantiago_FT2D_filtrada.pdf")


invX=ifft2(filtr[1])
plt.figure()
plt.imshow(abs(invX))
plt.show()
