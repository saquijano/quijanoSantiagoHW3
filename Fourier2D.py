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
shi=fftshift(trans)
grashi=np.abs(shi)
fgraf=np.log(grashi)

plt.figure()
plt.imshow(abs(fgraf),cmap='gray')
plt.title("Transformada de Fourier")
plt.savefig("quijanoSantiago_FT2D.pdf")
#fin

SS=1/(altura*base)


#filtrarla, informacion sale de aprenderpython.net/transformada-de-fourier/
trans2 = fft2(arbol)
shi2=fftshift(trans2)
def borrar(shi2,abj,arr,izq,der):
	for i in range(np.shape(shi2)[0]):
		for j in range(np.shape(shi2)[1]):
			if (i<arr and i>abj and j<der and j>izq):
				shi2[i,j]=0
	return shi2

def salvar(shi2,abj,arr,izq,der):
	for i in range(np.shape(shi2)[0]):
		for j in range(np.shape(shi2)[1]):
			if (i<arr and i>abj and j<der and j>izq):
				shi2[i,j]=shi2[i,j]
			else:
				shi2[i,j]=0
	return shi2

#shi3=salvar(shi2,0,256,120,136)
shi4=borrar(shi2,110,120,102,112)
shi5=borrar(shi4,130,140,146,156)
shi6=borrar(shi5,60,70,60,70)
shi7=borrar(shi6,190,200,190,200)
filGra=np.abs(shi7)
graficarFil=np.log(filGra)
filtra=ifftshift(shi7)
invX2=ifft2(filtra)
#

#f2=fftshift(filtr[0])
#graf2=np.log(np.abs(f2))
plt.figure()
plt.title("Transformada filtrada")
plt.imshow(graficarFil, cmap='gray')
plt.ylabel("frecuencia")
plt.xlabel(" ")
plt.savefig("quijanoSantiago_FT2D_filtrada.pdf")
#plt.show()

plt.figure()
plt.title("imagen despues de filtro")
plt.imshow(abs(invX2))
plt.show()
#######

#print(freqv,freqh)
#plt.figure()
#plt.title("Transformada de fourier")
#plt.xlabel("frecuencia")
#plt.plot(freqv,trans)

freqv=fftfreq(np.shape(trans)[0],SS)
freqh=fftfreq(np.shape(trans)[1],SS)

frecuencia=np.zeros((256,256))
for i in range(np.shape(frecuencia)[0]):
	frecuencia[:,i]=freqv

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

def bajosen(freq,sube):
	freqmenor=np.linspace(0,0,len(freq))
	lista=np.linspace(0,0,len(sube))
	for i in range(len(freq)):
		if (freq[i]<2400 and freq[i]>-2400):
			freqmenor[i]=freq[i]
			#lista[i]=sube[i]
	return freqmenor#, lista

filtr=bajos(frecuencia,trans)

##### no mover
#invX=ifft2(filtr[1])

#plt.figure()
#plt.title("imagen despues de filtro")
#plt.imshow(abs(invX))
#plt.show()
