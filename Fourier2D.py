import matplotlib.pylab as plt
#import cv2
import numpy as np
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2, fftshift, ifftshift


arbol=plt.imread("arbol.png")
plt.imshow(arbol)


#transformada
base,altura=np.shape(arbol)
trans = fft2(arbol)

shi=fftshift(trans)
grashi=np.abs(shi)
fgraf=np.log(grashi)

#grafica de la transformada
plt.figure()
plt.imshow(abs(fgraf), cmap='gray')
plt.title("Transformada de Fourier")
plt.savefig("quijanoSantiago_FT2D.pdf")


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
shi4=borrar(shi2,117,120,103,106)
shi5=borrar(shi4,136,139,151,154)
shi6=borrar(shi5,62,65,62,65)
shi7=borrar(shi6,191,194,191,194)
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

plt.figure()
plt.title("Imagen despues de filtro")
plt.imshow(abs(invX2))
plt.savefig("quijanoSantiago_Imagen_Filtrada.pdf")
#######

