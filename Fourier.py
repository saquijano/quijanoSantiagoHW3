import numpy as np
import matplotlib.pylab as plt
import scipy as sp
from scipy.fftpack import fft, fftfreq,ifft
from scipy import interpolate
import math

senal=np.genfromtxt("signal.dat",delimiter=",")

xsen=senal[:,0]
ysen=senal[:,1]

#grafica de la senal sin transformar
plt.figure()
plt.plot(xsen,ysen,label="Senal")
plt.xlabel("tiempo")
plt.ylabel("y(x)")
plt.title("Senal")
plt.savefig("quijanoSantiago_signal.pdf")

SS=xsen[1]-xsen[0] #sample spacing /dt
SR=1/SS #sample rating (1/dt)

def transfor(xsen,ysen):
	N=len(xsen)
	pi=np.pi
	sumar=np.linspace(0,0,len(xsen))
	for i in range(len(xsen)):
		for j in range(len(ysen)):
			sumar[i]+=ysen[j]*np.exp((-1j)*2*pi*i*j/N)
	sumar=sumar/N
	return sumar


sumar=abs(transfor(xsen,ysen))
sumar1=transfor(xsen,ysen)

def invrs(trans,xsen):
	sumar=np.linspace(0,0,len(xsen))
	for i in range(len(xsen)):
		for j in range(len(trans)):
			sumar[i]+=trans[j]*math.e**((+1j)*np.pi*2*j*i/512)
	return sumar

#uso los paquetes para recuperar las frecuencias
########
#fft_enX=fft(ysen)/len(xsen) #solo la hice para comparar
#graficar=abs(fft_enX)  #solo se usan valores positivos
freqx=fftfreq(len(xsen),SS) #frecuencias de funcion

#transformada
plt.figure()
plt.plot(freqx,sumar)
plt.title("Transformada de fourier")
plt.xlabel("Frecuencia")
plt.ylabel("Trans. Fourier")
plt.savefig("quijanoSantiago_TF.pdf")

#funcion que devuelve las frecuencias mas altas
def principales(freqx):
	guardo=[]
	hola=[]
	for i in range(len(freqx)):
		if sumar[i]>0.40:
			guardo.append(freqx[i])
			hola.append(i)
	return guardo

#frecuencias principales
print("Las frecuencias principales son:"+str(principales(freqx)))

##funcion para pasar solo los valores menores a 1000
def bajos(freq,sube):
	freqmenor=[]
	lista=[]
	for i in range(len(freq)):
		if (abs(freq[i])<=1000):
			freqmenor.append(freq[i])
			lista.append(sube[i])
	return freqmenor, lista

def bajos2(freq,sube):
	freqmenor=np.linspace(0,0,len(freq))
	lista=np.linspace(0,0,len(sube))
	for i in range(len(freq)):
		if (abs(freq[i])<=1000):
			freqmenor[i]=freq[i]
			lista[i]=sube[i]
	return freqmenor, lista

filtr=bajos(freqx,sumar1) ###sumar es transformada
filtr2=bajos2(freqx,sumar1)
# filtrada de fourier
#plt.figure()
#plt.plot(filtr2[0],filtr2[1], label="1")
#plt.plot(filtr[0],filtr[1],label="2")
#plt.title("Transformada de fourier filtrada")
#plt.xlabel("frecuencia")
#plt.legend()
#plt.ylabel("Trans. Fourier")

#trasnformada inversa uso los vaores de la transformada negativos y positivos realizo la inversa de la transformada de fourier
invX=ifft(filtr2[1])
invX42=ifft(sumar1)
inve=invrs(sumar1,xsen)
#inv=invrs(filtr2[1])
#creo linspace con mismo tiempo de la funcion original
tInv=np.linspace(min(xsen),max(xsen),len(invX))
#inversa de la filtrada de fourier
plt.figure()
plt.plot(xsen,invX)
plt.title("Transformada inversa de fourier")
plt.xlabel("tiempo")
plt.ylabel("y(x)")
plt.legend()
plt.savefig("quijanoSantiago_filtrada.pdf")

################################3
incompletos=np.genfromtxt("incompletos.dat",delimiter=",")

xinc=incompletos[:,0]
yinc=incompletos[:,1]

#parte datos incompletos
print("--------")
print("No se puede realizar la transformada de fourier porque los datos no tienen el mismo espaciento de tiempo. En si, se podria realizar la transformada pero la transformacin devuelve informacion falsa")


def interpolar (x,y):
	x2=np.linspace(min(x),max(x),512)
	cubi=sp.interpolate.interp1d(x,y,kind='cubic')
	cuadr=sp.interpolate.interp1d(x,y,kind='quadratic')
	cub1=cubi(x2)
	cua1=cuadr(x2)
	cuadra=sp.interpolate.splrep(x,y,k=2)
	cua=sp.interpolate.splev(x2,cuadra)
	cubica=sp.interpolate.splrep(x,y,k=3)
	cub=sp.interpolate.splev(x2,cubica)
	return cua,cub,cua1,cua1,x2

cua=interpolar(xinc,yinc)[0]
cua1=interpolar(xinc,yinc)[2]
cub=interpolar(xinc,yinc)[1]
cub1=interpolar(xinc,yinc)[3]
xinter=interpolar(xinc,yinc)[4]

#creo sample spaceing para las interpoladas
SS1=xinter[1]-xinter[0]
freqCub=fftfreq(len(xsen),SS1) #frecuencias de funcion cubica

tcub=abs(transfor(xsen,cub))
tcua=abs(transfor(xsen,cua))
freqCua=fftfreq(len(xsen),SS1) #frecuencias de funcion cuadrada

## transformdas de las interpolaciones
plt.figure()
plt.subplot(3,1,1)
plt.title("Las tres transformaciones de fourier")
plt.plot(freqCua,tcua,label="cuadrada",c="b")
plt.xlabel("frecuencia")
plt.ylabel("Trans. Fourier")
plt.legend()
plt.xlim(-1200,1200)
plt.subplot(3,1,2)
plt.plot(freqx,sumar,label="completa",c="y")
plt.xlabel("frecuencia")
plt.ylabel("Trans. Fourier")
plt.legend()
plt.xlim(-1200,1200)
plt.subplot(3,1,3)
plt.plot(freqCub,tcub,label="cubica",c="g")
plt.xlabel("frecuencia")
plt.ylabel("Trans. Fourier")
plt.legend()
plt.xlim(-1200,1200)
plt.savefig("quijanoSantiago_TF_interpola.pdf")

print("-----")
print("No se observan grandes diferencias entre los valores obtenidos en la transformadas de fourier de las tres frecuencia principales. Sin embargo la frecuencia de 480 tiene menor valor en las interpolaciones. La informacion, aunque se recupera parcialmente al interpolar, no se recupera toda. Hay mas ruido en frecuencias amas importante se ve una disminucion importante, infromacion que se ha perdido. Las otras frecuencias parecen comportarse de manera similar, incluyendo aquella entre las frecuencias principales. La frecuencias en las transformaciones interpoladas parecen concentrarse entre -2500 y 2500, mientra la original tiene frecuencias que van de -10000 a 10000.")

#funcion para pasar solo los valores menores a 1000
#def bajos500(freq,sube):
#	freqmenor=[]
#	lista=[]
#	for i in range(len(freq)):
#		if (freq[i]<500 and freq[i]>-500):
#			freqmenor.append(freq[i])
#			lista.append(sube[i])
#	return freqmenor, lista

def bajos500(freq,sube):
	freqmenor=np.linspace(0,0,len(freq))
	lista=np.linspace(0,0,len(sube))
	for i in range(len(freq)):
		if (abs(freq[i])<=1000):
			freqmenor[i]=freq[i]
			lista[i]=sube[i]
	return freqmenor, lista

# funciones interpoladas transformadas
tcub2=transfor(xsen,cub)
tcua2=transfor(xsen,cua)
#Se le aplica el filtro a las funciones transformadas
cub500=bajos500(freqCub,tcub2)
cub1000=bajos2(freqCub,tcub2)
cua500=bajos500(freqCua,tcua2)
cua1000=bajos2(freqCua,tcua2)
ori500=bajos500(freqx,sumar1)
filtr=bajos2(freqx,sumar1)
#Inversa de las funciones filtradas
icub500=invrs(cub500[1],xsen)
icub1000=invrs(cub500[1],xsen)
icua500=invrs(cua500[1],xsen)
icua1000=invrs(cua1000[1],xsen)
iori500=invrs(ori500[1],xsen)
iori1000=invrs(filtr2[1],xsen)


#filtro funciones transformadas filtras
#plt.figure()
#plt.subplot(2,1,1)
#plt.plot(cua500[0],cua500[1],label="500 cuadratico")
#plt.plot(cub500[0],cub500[1],label="500 cubico")
#plt.plot(ori500[0],ori500[1],label="500 originales")
#plt.xlabel("tiempo")
#plt.ylabel("frecuencia")
#plt.legend()
#plt.title("Filtros 500")
#plt.subplot(2,1,2)
#plt.plot(cua1000[0],cua1000[1],label="500 cuadratico")
#plt.plot(cub1000[0],cub1000[1],label="1000 cubico")
#plt.plot(filtr[0],filtr[1],label="1000 originales")
#plt.xlabel("tiempo")
#plt.ylabel("frecuencia")
#plt.title("Filtros 1000")
#plt.legend()


#####
#funciones inversas filtradas
plt.figure()
plt.subplot(2,1,1)
plt.plot(xsen,icub500,label="500 cubico")
plt.plot(xsen,iori500,label="500 originales")
plt.plot(xsen,icua500,label="500 cuadratico")
plt.xlabel("tiempo")
plt.ylabel("y(t)")
plt.legend()
plt.title("Senales filtradas")
plt.subplot(2,1,2)
plt.plot(xsen,icub1000,label="1000 cuadratico")
plt.plot(xsen,iori1000,label="1000 originales")
plt.plot(xsen,icua1000,label="1000 cubico")
plt.xlabel("tiempo")
plt.ylabel("y(t)")
plt.legend()
plt.savefig("quijanoSantiago_2Filtros.pdf")

