import matplotlib.pylab as plt
#import cv2
import numpy as np
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2, fftshift, ifftshift
from scipy import misc
from skimage import io


arbol=sp.misc.imread("arbol.png",mode='L')
plt.imshow(arbol)


#transformada
base,altura=np.shape(arbol)
trans = fft2(arbol)
#nuevo
f=fftshift(trans)
graf=np.abs(f)
fgraf=np.log(graf)


plt.figure()
plt.imshow(abs(fgraf),cmap='gray')
plt.title("intento")
#fin
SS=1/(altura*base)



freqv=fftfreq(np.shape(trans)[0],SS)
freqh=fftfreq(np.shape(trans)[1],SS)
print(freqv,freqh)
plt.figure()
plt.title("Transformada de fourier")
plt.xlabel("frecuencia")
plt.plot(freqv,trans)
plt.legend()

graf=abs(trans)


#funcion para filtrar transformada
def bajos(freq,sube):
	freqmenor=np.zeros((256,256))
	lista=np.zeros((256,256))
	for i in range(np.shape(sube)[0]):
		for j in range(np.shape(sube)[1]):
			if ((freq[i,j]>2000 or freq[i,j]<-2000) or (freq[i,j]<500 and freq[i,j]>-500)):
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
		if (freq[i]<2400 and freq[i]>-2400):
			freqmenor[i]=freq[i]
			#lista[i]=sube[i]
	return freqmenor#, lista


filtr=bajos(frecuencia,trans)
print(trans)

f=fftshift(trans)
graf=np.abs(f)
fgraf=np.log(np.abs(f))

#filtrarla, informacion sle de aprenderpython.net/transformada-de-fourier

f[-120:,-40:-10]=0
graFil=np.abs(f)
fFiltrada=np.log(graFil)
filtra=ifftshift(f)
invX2=ifft2(abs(filtra))
#

#f2=fftshift(filtr[0])
#graf2=np.log(np.abs(f2))
plt.figure()
plt.title("Transformada filtrada")
plt.imshow(fFiltrada, dmap='gray')
plt.ylabel("frecuencia")
plt.xlabel(" ")
plt.savefig("quijanoSantiago_FT2D_filtrada.pdf")
#plt.show()

plt.figure()
plt.title("imagen despues de filtro")
plt.imshow(abs(invX2))
plt.show()
#######




##### no mover
#invX=ifft2(filtr[1])

#plt.figure()
#plt.title("imagen despues de filtro")
#plt.imshow(abs(invX))
#plt.show()
