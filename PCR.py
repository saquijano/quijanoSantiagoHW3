import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import *

#uso 
preli=np.genfromtxt("WDBC.dat",delimiter=",",dtype='U16')
filas=np.shape(preli)[0]
colum=np.shape(preli)[1]
datos=np.zeros((filas,colum-2))
datosMal=np.zeros((filas,colum))
#numeros los meti a mano despues de contarlos con k y w
benignos=np.zeros((357,np.shape(datos)[1]))
malignos=np.zeros((212,np.shape(datos)[1]))

hola1=preli[:,3]
dondeB=[]
dondeM=[]
#converti la segunda columna en binaria. M =0 B=1
k=0
w=0
for i in range(np.shape(preli)[0]):
	for j in range(np.shape(preli)[1]):
		if (j==0):
			datosMal[i,j]=float(preli[i,j])
		elif (j==1):
			if (preli[i,j]=='B'):
				datosMal[i,j]=1
				benignos[k,:]=preli[i,j+1:]
				k=k+1
			else:
				datosMal[i,j]=0
				malignos[w,:]=preli[i,j+1:]
				w=w+1
		else:
			datosMal[i,j]=float(preli[i,j])
			datos[i,j-2]=float(preli[i,j])
	
#print(preli[:,1])
#print(malignos)
#print(w,k)
#print(len(malignos))

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

# no lo uso, es evidente cuales son
def factoresPri(valores,vectores):
	temporal=0
	temporal2=0
	for i in range(len(valores)):
		if (temporal<valores[i] and temporal2<valores[i]):
			temporal=valores[i]
		elif (temporal2<valores[i] and temporal<valores[i]):
			temporal2=valores[i]		
	return temporal,temporal2

print("valor propio principal, el 1:", valores[0])
print("Con vector propio:", vectores[:,0])

print("vectores propio secundario, el 2", valores[1])
print("Con vector propio:", vectores[:,1])

pEje=np.matmul(datos,vectores)
benPlot=np.matmul(benignos,vectores)
#benPlot2=np.matmul(benignos,vectores[1])

malPlot=np.matmul(malignos,vectores)
#malPlot2=np.matmul(malignos,vectores[1])
princi=np.matmul(datos,vectores[2])

PC1=pEje[:,0]
PC2=pEje[:,1]

print ("el eje principal es:",PC1)

print ("El eje secundario es:", PC2)
plt.figure()
plt.title("PCA, primeros componentes")
plt.scatter(benPlot[:,0],benPlot[:,1],label="benignos")
plt.scatter(malPlot[:,0],malPlot[:,1],label="malignos")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(princi)
plt.title("otro")


print("El metodo de pca ayuda a reducir las variables que se usan para analizar un problema. Por esto podria ser util en este caso, ya que, se hace un estudio con muchas variables. Sin embargo no parce que se separen mucho los grupos de beningno y maligno. Por lo cual puede que estos dos ejes no ayuden a clarificar el tipo de tumor.")
