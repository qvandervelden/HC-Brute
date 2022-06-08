import numpy as np
import numpy.linalg as lin
import time

start = time.time() #Starting the timer for the runtime

J = -1 #The coupling between neightbouring spins
h = 1 #The reduced plank constant
N = 12 #The number of particles

#the identity
I = np.array([[1,0],[0,1]])

#The spin-operators
Sx = h/2*np.array([[0,1],[1,0]])
Sy = h/2*np.array([[0,-1j],[1j,0]])
Sz = h/2*np.array([[1,0],[0,-1]])

#Computation of the tensor products of the spin-operators
S = np.empty(shape=(0,3,2**N,2**N))
for i in range(0,N):
    Sx_i = Sx
    Sy_i = Sy
    Sz_i = Sz
    for j in range(0,i):
        Sx_i = np.kron(I,Sx_i)
        Sy_i = np.kron(I,Sy_i)
        Sz_i = np.kron(I,Sz_i)
    for j in range(i+1,N):
        Sx_i = np.kron(Sx_i, I)
        Sy_i = np.kron(Sy_i, I)
        Sz_i = np.kron(Sz_i, I)
    S_i = np.array([[Sx_i, Sy_i, Sz_i]])
    S = np.concatenate((S, S_i))
    
#Computation of the matrix for the Hamiltonian
H = J*(np.matmul(S[0,0], S[1,0]) + np.matmul(S[0,1], S[1,1]) + np.matmul(S[0,2], S[1,2]))
for i in range(1,N-1):
    H = H + J*(np.matmul(S[i,0], S[i+1,0]) + np.matmul(S[i,1], S[i+1,1]) + np.matmul(S[i,2], S[i+1,2]))
# H = H + J*(np.matmul(S[N-1,0], S[0,0]) + np.matmul(S[N-1,1], S[0,1]) + np.matmul(S[N-1,2], S[0,2])) #Final term for the circle

#Computation of the eigenenergies
w = lin.eigvalsh(H)
idx = w.argsort()[::1]
w = w[idx]
print(w)

print(time.time() - start) #Printing runtime