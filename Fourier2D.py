import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy import misc
from skimage import io


arbol=sp.misc.imread("arbol.png",mode='RGB',flatten=True)
#arbol2=io.read("arbol.png")/255
print(type(arbol))
plt.imshow(arbol)
plt.show()

#transformada

trans = np.fft.fft2(arbol)
plt.plot(trans)
plt.show()

