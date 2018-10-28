import numpy as np
import matplotlib.pylab as plt
import scipy as sp
from scipy.fftpack import fft, fftfreq,ifft
import scipy.io.wavfile as wav
from scipy import interpolate

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
#plt.show()

SS=xsen[1]-xsen[0] #sample spacing /dt

SR=1/SS #sample rating (1/dt)

# no se uso
def transfor(xsen,ysen):
	sumar=np.linspace(0,0,len(xsen))
	for i in range(len(xsen)):
		for j in range(len(xsen)):
			sumar[i]=sumar[i]+(np.exp(-1j*2*np.pi*j*i/512)*ysen[j])
	sumar=sumar/512
	return sumar
sumar=abs(transfor(xsen,ysen))
sumar1=transfor(xsen,ysen)

#uso los paquetes para recuperar las frecuencias
########
#fft_enX=fft(ysen)/len(xsen) #solo la hice para comparar
#graficar=abs(fft_enX)  #solo se usan valores positivos
#######
freqx=fftfreq(len(xsen),SS) #frecuencias de funcion

#transformada
plt.figure()
plt.plot(freqx,sumar)
plt.title("Transformada de fourier")
plt.xlabel("frecuencia")
plt.ylabel("")
plt.savefig("quijanoSantiago_TF.pdf")
#plt.show()


#funcion que devuelve las frecuencias mas altas

def principales(freqx):
	guardo=[]
	hola=[]
	for i in range(len(freqx)):
		if sumar[i]>0.45:
			guardo.append(freqx[i])
			hola.append(i)
	return guardo


#frecuencias principales
print("Las frecuencias principales son:"+str(principales(freqx)))

#funcion para pasar solo los valores menores a 1000
def bajos(freq,sube):
	freqmenor=[]
	lista=[]
	for i in range(len(freq)):
		if (freq[i]<1000 and freq[i]>-1000):
			freqmenor.append(freq[i])
			lista.append(sube[i])
	return freqmenor, lista

def bajos2(freq,sube):
	freqmenor=np.linspace(0,0,len(freq))
	lista=np.linspace(0,0,len(sube))
	for i in range(len(freq)):
		if (freq[i]<1000 and freq[i]>-1000):
			freqmenor[i]=freq[i]
			lista[i]=sube[i]
	return freqmenor, lista

filtr=bajos(freqx,sumar1)
filtr2=bajos2(freqx,sumar1)

plt.figure()
plt.plot(filtr2[0],filtr2[1])
plt.plot(filtr[0],filtr[1])
plt.title("Transformada de fourier filtrada")
plt.xlabel("frecuencia")
plt.ylabel("")
#plt.savefig("quijanoSantiago_filtrada.pdf")
#plt.show()



#################################
#trasnformada inversa uso los vaores de la transformada negativos y positivos


mismoTam=len(xsen)/len(filtr[1])
#realizo la inversa de la transformada de fourier
invX=ifft(filtr2[1])

#creo linspace con mismo tiempo de la funcion original
tInv=np.linspace(min(xsen),max(xsen),len(invX))

plt.figure()
plt.plot(xsen,invX)
plt.title("Transformada inversa de fourier")
plt.xlabel("tiempo")
plt.ylabel("y(x)")
plt.savefig("quijanoSantiago_filtrada.pdf")
plt.show()

################################3
incompletos=np.genfromtxt("incompletos.dat",delimiter=",")

xinc=incompletos[:,0]
yinc=incompletos[:,1]

#parte datos incompletos

print(len(xinc),len(xsen))
print("A los dator incompletos si se le puede hacer una transformada de fourier pero parece que esta funcion tiene menos frecuencias, ya que tiene menos ruido. Por esto no tiene sentido realizar una transformada de fourier")

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

tcub=np.linspace(0,0,len(cub))
for i in range(len(xsen)):
	for j in range(len(xsen)):
		tcub[i]=tcub[i]+(np.exp(-1j*2*np.pi*j*i/512)*cub[j])
tcub=abs(tcub/512)

freqCub=fftfreq(len(xsen),SS1) #frecuencias de funcion cubica


tcua=np.linspace(0,0,len(cub))
for i in range(len(xsen)):
	for j in range(len(xsen)):
		tcua[i]=tcua[i]+(np.exp(-1j*2*np.pi*j*i/512)*cua[j])
tcua=abs(tcua/512)
freqCua=fftfreq(len(xsen),SS1) #frecuencias de funcion cuadrada

print(SS,SS1,len(xinter),len(xsen))

plt.figure()
plt.plot(freqCua,tcua,label="cuadrada")
#plt.plot(cua1,label="2")
plt.plot(freqx,sumar,label="completa")
plt.plot(freqCub,tcub,label="cubica")
#plt.plot(cub1,label="4")
plt.xlabel("frecuencia")
plt.ylabel("y(x)")
plt.title("tres transformadas de fourier")
plt.legend()
plt.savefig("quijanoSantiago_TF_interpola.pdf")
#plt.show()


print("No se observan grandes diferencias entre las transformadas de fourier de la frecuencia principal. Sin embargo para la segunda frecuencia mas importante se ve una disminucion importante que se ha perdido. Las otras frecuencias parecen comportarse de manera similar, incluyendo aquella entre las frecuencias principales.")

#funcion para pasar solo los valores menores a 1000
def bajos500(freq,sube):
	freqmenor=[]
	lista=[]
	for i in range(len(freq)):
		if (freq[i]<500 and freq[i]>-500):
			freqmenor.append(freq[i])
			lista.append(sube[i])
	return freqmenor, lista

cub500=bajos500(freqCub,tcub)
cub1000=bajos(freqCub,tcub)
cua500=bajos500(freqCua,tcua)
cua1000=bajos(freqCua,tcua)
ori500=bajos500(freqx,sumar)
filtr=bajos(freqx,sumar)

#filtro
plt.figure()
plt.plot(cub500[0],cub500[1],label="500 cubico")
plt.plot(cub[0],cub[1],label="1000 cubico")
plt.plot(cua500[0],cua500[1],label="500 cuadratico")
plt.plot(cua[0],cua[1],label="500 cuadratico")
plt.plot(ori500[0],ori500[1],label="500 originales")
plt.plot(filtr[0],filtr[1],label="1000 originales")
plt.xlabel("tiempo")
plt.ylabel("frecuencia")
plt.legend()
plt.title("Filtros de todas las graficas")
plt.savefig("quijanoSantiago_2Filtros.pdf")
#plt.show()
