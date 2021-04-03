import numpy as np
import math

def traces(a,p,q,T,boolean):

    K_range=np.arange(3,21,1)
    Kmax=K_range[-1]

    A_k=np.array([])
    Q_k=np.array([])
    P_t=np.array([])

    if boolean==0:
        for k in range(Kmax+Kmax*T):

            a_k = np.random.uniform(0,a)

            q_k = np.random.uniform(0,q)

            A_k = np.append(A_k, a_k)

            Q_k = np.append(Q_k, q_k)


        for t in range(T+1):

            p_t = np.random.uniform(0,p)

            P_t = np.append(P_t, p_t)
    else:
        
        for k in range(Kmax+Kmax*T):

            q_k = np.sin(k*math.pi/5) + np.random.uniform(1,q)

            Q_k = np.append(Q_k, q_k)

            a_k = np.sin(k*math.pi/5) + np.random.uniform(1,a)

            A_k = np.append(A_k, a_k)

        for t in range(T+1):

            p_t = np.sin(math.pi*t/5) + np.random.uniform(1,p)

            P_t = np.append(P_t, p_t)
            
    return A_k, P_t, Q_k
