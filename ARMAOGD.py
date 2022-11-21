##### prediction module
import numpy as np
def onlineOGD(m,k,eta,c,serie):
    #m,k=1,1
    q=m+k
    #parameters
    #c=1
    #eta=10

    #initialization
    Predictions=np.array([])
    #loss=0
    #Regret=np.array([])
    #Relative_error=np.array([])
    
    gamma_prev = np.random.uniform(-c,c,q)

    Gamma = np.zeros((len(serie)-q, q))


    for t in range(q,len(serie)):

        x_prev = serie[t-q:t]

        x_prev = list(x_prev)
        x_prev.reverse()

        prediction = np.dot(gamma_prev,x_prev)
        value=serie[t]

        #store the predictions
        Predictions = np.append(Predictions, prediction)
        
        #store the regret values
        #loss += (value-prediction)**2
        #Regret = np.append(Regret, loss/t)
        #Relative_error = np.append(Relative_error, abs(abs(value - prediction)/value))


        #gradient descent update
        for i in range(q):
            gamma_prev[i] = gamma_prev[i] - (1/eta)*2*(value - prediction)*(-1)*x_prev[i]
            
            #projection step onto convex set K
            gamma_prev[i] = min(max(gamma_prev[i],-1),1)

        Gamma[t-q] = gamma_prev
        
    return Predictions#, Regret, Relative_error
