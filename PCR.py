import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import *

#uso 
preli=np.genfromtxt("WDBC.dat",delimiter=",",dtype='U16')
filas=np.shape(preli)[0]
colum=np.shape(preli)[1]
datos=np.zeros((filas,colum-1))
datosSexo=np.zeros((filas,colum))
hola1=preli[:,3]

#converti la segunda columna en binaria. M =0 B=1

for i in range(np.shape(preli)[0]):
	for j in range(np.shape(preli)[1]):
		if (j==0):
			datosSexo[i,j]=float(preli[i,j])
			datos[i,j]=int(preli[i,j])
		elif (j==1):
			if (preli[i,j]=='B'):
				datosSexo[i,j]=1
			else:
				datosSexo[i,j]=0
		else:
			datosSexo[i,j]=float(preli[i,j])
			datos[i,j-1]=float(preli[i,j])

print(np.shape(datos)[0],np.shape(datos)[1])

hola=np.array([[1,2,1], [4,2,13],[7,8,1], [8,4,5]])
chao=np.array([3,4])


def matrizCov(datos):
	variables=np.shape(datos)[1]
	cantidad=np.shape(datos)[0]
	matriz=np.zeros((variables,variables))
	for i in range(variables):
		prom1=np.mean(datos[:,i])
		vec=datos[:,i]-prom1
		for j in range(i,variables):
			temp2=np.mean(datos[:,j])
			vec2=datos[:,j]-temp2
			arriba=np.sum(vec*vec2)
			abajo=cantidad
			matriz[i,j]=arriba/abajo
			matriz[j,i]=arriba/abajo
	return matriz


matriz=matrizCov(datos)
matriz2=np.cov(datos)
#cero=matriz-matriz2
valores=np.linalg.eig(matriz)[0]
vectores=np.linalg.eig(matriz)[1]
def factoresPri(valores,vectores):
	temporal=0
	temporal2=0
	for i in range(len(valores)):
		if (temporal<valores[i] and temporal2<valores[i]):
			temporal=valores[i]
		elif (temporal2<valores[i] and temporal<valores[i]):
			temporal2=valores[i]		
	return temporal,temporal2


#print("valor propio principal, el 3:", valores[2])
print("Con vector propio:", vectores[:,0])

#print("vectores propio secundario, el 2", valores[1])
print("Con vector propio:", vectores[:,1])

pEje=np.matmul(datos,vectores[0])
sEje=np.matmul(datos,vectores[1])

plt.scatter(pEje,sEje)
plt.show()


print("El metodo de pca ayuda a reducir las variables que se usan para analizar un problema. Por esto es util en este caso, ya que, reduce la dimensionalidad de la variables pero permite el analisis del problema con estas")
