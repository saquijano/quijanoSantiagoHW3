import numpy as np
import matplotlib.pylab as plt
import scipy as sp
from scipy.fftpack import fft, fftfreq
import scipy.io.wavfile as wav

senal=np.genfromtxt("signal.dat",delimiter=",")

xsen=senal[:,0]
ysen=senal[:,1]

SS=xsen[1]-xsen[0] #sample spacing

SR=1/SS #sample rating

sumar=np.linspace(0,0,len(xsen))
for i in range(len(xsen)):
	for j in range(len(t)):
		sumar[i]=sumar[i]+(np.exp(-1j*2*np.pi*j*i/128)*y[j])

#fft_enX=fft(ysen)/len(xsen)
#graficar=abs(fft_enX)
#freqx=fftfreq(len(xsen),SS)
#plt.plot(freqx,graficar)
#plt.show()

plt.figure()
plt.plot(xsen,ysen,label="Senal")
plt.xlabel("tiempo")
plt.ylabel("frecuencia")
plt.title("Senal")
plt.savefig("quijanoSantiago_signal.pdf")

print (freqx)

incompletos=np.genfromtxt("incompletos.dat",delimiter=",")

xinc=incompletos[:,0]
yinc=incompletos[:,1]


plt.figure()
plt.plot(xinc,yinc,label="Senal")
plt.xlabel("tiempo")
plt.ylabel("frecuencia")
plt.title("Datos incompletos")
#plt.savefig("quijanoSantiago_signal.pdf")
