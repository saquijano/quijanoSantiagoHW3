import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import *

#uso 
preli=np.genfromtxt("WDBC.dat",delimiter=",",dtype='U16')
filas=np.shape(preli)[0]
colum=np.shape(preli)[1]
datos=np.zeros((filas,colum-2))
datosTod=np.zeros((filas,colum))
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
			datosTod[i,j]=float(preli[i,j])
		elif (j==1):
			if (preli[i,j]=='B'):
				datosTod[i,j]=1
				benignos[k,:]=preli[i,j+1:]
				k=k+1
			else:
				datosTod[i,j]=0
				malignos[w,:]=preli[i,j+1:]
				w=w+1
		else:
			datos[i,j-2]=float(preli[i,j])
			datosTod



def norma(datos,benignos,malignos):
	for i in range(np.shape(datos)[1]):
		prome=np.mean(datos[:,i])
		desv=np.std(datos[:,i])
		datos[:,i]=(datos[:,i]-prome)/desv
		benignos[:,i]=(benignos[:,i]-prome)/desv
		malignos[:,i]=(malignos[:,i]-prome)/desv	
	return datos, benignos, malignos

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


datosNor=norma(datos,benignos,malignos)[0]
benigNor=norma(datos,benignos,malignos)[1]
maligNor=norma(datos,benignos,malignos)[2]

matriz=matrizCov(datosNor)

valores=np.linalg.eig(matriz)[0]
vectores=np.linalg.eig(matriz)[1]


print("valor propio principal, el 1:", valores[0])
print("Con vector propio:", vectores[:,0])

print("vectores propio secundario, el 2", valores[1])
print("Con vector propio:", vectores[:,1])

pEje=np.matmul(datosNor,vectores)
print(np.shape(pEje))
benPlot=np.matmul(benigNor,vectores)
#benPlot2=np.matmul(benignos,vectores[1])

malPlot=np.matmul(maligNor,vectores)
#malPlot2=np.matmul(malignos,vectores[1])
#princi=np.matmul(datos,vectores[2])

PC1=pEje[:,0]
PC2=pEje[:,1]

print ("el eje principal es:",PC1)

print ("El eje secundario es:", PC2)
plt.figure()
plt.title("PCA, componentes principales")
plt.scatter(benPlot[:,0],benPlot[:,1],label="benignos")
plt.scatter(malPlot[:,0],malPlot[:,1],label="malignos")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.savefig("quijanoSantiago_PCA.pdf")



print("El metodo de pca es util en este caso porque ademas de reducir las variables de estudio permite identificar algunos tumores malignos. Si el paciente muestra un valor elevado en el componente principal (PC1) y un valor negativo en el componente secundario (PC2) posiblemente sea un tumor maligno. Por otra parte si presenta valores negativos en PC1 y positivos de PC2, posiblemente sea un tumor benigno. En los otros dos cuadrantes presentan un division entre maligno donde se observa que valores mas cercanos a 0 de PC1 y positvos de PC2 van a asociarse con tumores benignos y valores negativos de PC2 con valores negativos cercanos a 0 de PC1 se asocian con tumores malignos.")
