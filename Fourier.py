import numpy as np
import matplotlib.pylab as plt
import scipy as sp
from scipy.fftpack import fft, fftfreq,ifft
import scipy.io.wavfile as wav

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
sumar=np.linspace(0,0,len(xsen))
for i in range(len(xsen)):
	for j in range(len(xsen)):
		sumar[i]=sumar[i]+(np.exp(-1j*2*np.pi*j*i/128)*ysen[j])

#uso los paquetes para recuperar las frecuencias

fft_enX=fft(ysen)/len(xsen)
graficar=abs(fft_enX)  #solo se usan valores positivos
freqx=fftfreq(len(xsen),SS) #frecuencias de funcion

#transformada
plt.figure()
plt.plot(freqx,fft_enX)
#plt.scatter(abs(sumar),fft_enX)
plt.title("Transformada de fourier")
plt.xlabel("frecuencia")
plt.ylabel("")
plt.savefig("quijanoSantiago_TF.pdf")
#plt.show()


#creo dos listas una para conocer la posicion de las frecuencias mas altas y la otra para guardar el valor para la frecuencia
guardo=[]
hola=[]
for i in range(len(freqx)):
	if fft_enX[i]>0.4:
		guardo.append(freqx[i])
		hola.append(i)

print(graficar[0],hola)


#frecuencias principales
print("Las frecuencias principales son:"+str(guardo))

#funcion para pasar solo los valores menores a 1000
def bajos(freq,sube):
	freqmenor=[]
	lista=[]
	for i in range(len(freq)):
		if (freq[i]<1000 and freq[i]>-1000):
			freqmenor.append(freq[i])
			lista.append(sube[i])
	return freqmenor, lista

filtr=bajos(freqx,fft_enX)

plt.figure()
plt.plot(filtr[0],filtr[1])
plt.title("Transformada de fourier filtrada")
plt.xlabel("frecuencia")
plt.ylabel("")
plt.savefig("quijanoSantiago_filtrada.pdf")
#plt.show()

incompletos=np.genfromtxt("incompletos.dat",delimiter=",")

mismoTam=len(xsen)/len(filtr[1])
#realizo la inversa de la transformada de fourier
invX=ifft(filtr[1])

#creo linspace con mismo tiempo de la funcion original
tInv=np.linspace(min(xsen),max(xsen),len(invX))

plt.figure()
plt.plot(tInv,invX)
plt.title("Transformada inversa de fourier")
plt.xlabel("tiempo")
plt.ylabel("y(x)")
plt.show()

def invertir(freq,valor):
	freqMcero=[]
	rela=[]
	for i in range(len(valor)):
		if (freq[i]>=0):
			freqMcero.append(freq[i])
			rela.append(valor[i])
	return 12

xinc=incompletos[:,0]
yinc=incompletos[:,1]

#parte datos incompletos

print(len(xinc),len(xsen))
print("A los dator incompletos si se le puede hacer una transformada de fourier pero parece que esta funcion tiene menos frecuencias, ya que tiene menos ruido. Por esto no tiene sentido realizar una transformada de fourier")

def interpolar (x,y):
	x2=np.linspace(min(xi),max(xi),512)
	cubi=sp.interpolate.interp1d(x,y,'cubic')
	cuadr=sp.interpolate.interp1d(x,y,'quadratic')
	cub1=cubi(x2)
	cua1=cuadr(x2)
	cuadra=sp.interpolate.splrep(x,y,k=2)
	cua=sp.interpolate.splev(x2,cuadra)
	cubica=sp.interpolate.splrep(x,y,k=3)
	cub=sp.interpolate.splev(x2,cubica)
	return cua,cub,cua1,cua1
cua=interpolar(xinc,yinc)[0]
cua1=interpolar(xinc,yinc)[2]
cub=interpolar(xinc,yinc)[1]
cub1=interpolar(xinc,yinc)[3]
plt.savefig()
plt.plot(cua)
plt.plot(cua1)
plt.show()
SS1=xinc[1]-xinc[0] #sample spacing /dt

SR1=1/SS1 #sample rating (1/dt)
fft_incX=fft(yinc)/len(xinc)
graficarInc=abs(fft_incX)  #solo se usan valores positivos
freqx1=fftfreq(len(xinc),SS1) #frecuencias de funcion

#no toca guardarlo
plt.figure()
plt.plot(freqx1,fft_incX,label="Senal")
plt.xlabel("tiempo")
plt.ylabel("frecuencia")
plt.title("Datos incompletos")
#plt.savefig("quijanoSantiago_signal.pdf")
plt.show()
