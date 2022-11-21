from scipy.optimize import minimize, Bounds
import numpy as np
import math

#calibr_option=0, alpha=0.6, facteur_pref=1, sigma=1, beta=0.3, option='last-in', lags=2
def simulationPred(start,a,p,q,theta,T,m,capacity_constraint, V, sigma,lags,c):
    
    #useful functions
    
    
    def onlinePred(m,k,eta,c,lags,x_prev,gamma_prev, prediction, value):
            
        #x_prev = serie[t-lags:t]

        #x_prev = list(x_prev)
        #x_prev.reverse()

        #prediction = np.dot(gamma_prev,x_prev)
        #value=serie[t]

        #gradient descent update
        for i in range(lags):
            gamma_prev[i] = gamma_prev[i] - (1/eta)*2*(value - prediction)*(-1)*x_prev[i]

            #projection step onto convex set K
            gamma_prev[i] = min(max(gamma_prev[i],-1),1)

        return gamma_prev

    def sqeuclidean(x):
        return np.dot(x,x)
    #objective function
    def f(x):
        return -V*a_t*np.log(1 + np.dot(x[:m],theta_t) + np.dot(x[m:],theta_t) ) + (np.dot(x[:m],p_t) + np.dot(x[m:],q_t))
    #gradient function
    def grad(x):
        output = np.zeros(2*m)
        for l in range(m):
            output[l]  = -V*a_t*theta_t[l]/(1 + np.dot(x[:m],theta_t) + np.dot(x[m:],theta_t)) + p_t[l]
        for l in range(m):
            output[l+m] = -V*a_t*theta_t[l]/(1 + np.dot(x[:m],theta_t) + np.dot(x[m:],theta_t)) + q_t[l]
        return output


    def reg(x):
        output = 0
        for i in range(t):
            output += sqeuclidean(x-Xprev[i])*sigma*(math.sqrt(H[i])-math.sqrt(H[i-1]))/2

        return output



    def acc_grad(x):

        return np.dot(acc_Grad[t-1],x)

    # time slot t will be defined as a global variable, not an argument    
    def toMinimize(x):
        #return reg(x) + acc_grad(x) + grad_pred(x)
        return reg(x) + acc_grad(x) + np.dot(Pred[t],x)

    def MinEstim(x):
        return reg(x) + acc_grad(x)
    

    #initialize performance vector
    loss=0
    loss_vec=np.array([])


    #initialize x1 as you wish (will do randomly)
    x1 = np.ones(2*m)
    Xprev = np.zeros((T+1,2*m))
    Xprev[0] = x1
    
    Xpred = np.zeros((T+1,2*m))
    Xpred[0] = x1

    #initialize last-in decision for the loop
    x_prev = x1

    #intialize traces
    a_t = a[0]
    theta_t = theta[0]
    p_t = p[0]
    q_t = q[0]

    #initialize gradient 
    grad1 = grad(x1)
    grad_serie = np.zeros((T+1,2*m))
    grad_serie[0] = grad1
    acc_Grad = np.zeros((T+1,2*m))
    acc_Grad[0] = grad1

    #initialize prediction on the gradient
    Pred = np.zeros((T+1,2*m))
    
   
    Gamma_prev = [np.random.uniform(-c,c,lags) for i in range(2*m)]
    
    #grad_serie[t-lags+k][i]
    #Grad_prev = [ for i in range(2*m)]
    Grad_prev = np.zeros((2*m,lags))

    #initialize error history
    H = np.zeros(T+1)
    H[0] = sqeuclidean(grad1-Pred[0])
    #Delta = np.zeros(T)
    Er = np.zeros(T)

    t=1

    while t<=T:
        #we start the prediction after decision of at least lags(=2) slots
        if t >=lags:
            for i in range(2*m):
                
                grad_prev = np.array([])
                for k in range(lags):
                    grad_prev = np.append(grad_prev, grad_serie[t-lags+k][i])
                    
                grad_prev = list(grad_prev)
                grad_prev.reverse()
                Grad_prev[i] = grad_prev
                
                    
                Pred[t][i] = np.dot(Grad_prev[i],Gamma_prev[i])




        my_lb=np.zeros(2*m)
        my_ub=np.array(capacity_constraint*2)
        my_bounds=Bounds(my_lb,my_ub)
        res=minimize(toMinimize, np.zeros(2*m), bounds=my_bounds)
        x_prev=res.x
        
        

        #update the history of decision
        Xprev[t] = x_prev
        #different between estimated next reservation and next reservation
        #Delta[t-1] = sqeuclidean(Xprev[t]-Xpred[t])
        #Delta[t-1] = sqeuclidean(Xprev[t]-Xprev[t-1])
        
        #update to current traces
        #(t+1)~current values of the traces
        a_t = a[t]
        theta_t = theta[t]
        p_t = p[t]
        q_t = q[t]
        
        
        #update gradient history
        acc_Grad[t] = acc_Grad[t-1]+grad(x_prev)
        grad_serie[t] = grad(x_prev)
        
        #update lag coefficients
        if t >= lags:
            for i in range(2*m):
                Gamma_prev[i] = onlinePred(m=1,k=1,eta=20,c=1,lags=lags, x_prev=Grad_prev[i], gamma_prev=Gamma_prev[i], prediction=Pred[t][i], value=grad_serie[t][i])
                

        #update error history (error is acumulated), just withdraw H[t-1] if you wanna look at error per slot
        H[t] = H[t-1] + sqeuclidean(grad(x_prev)-Pred[t])    
        #relative percentage error history
        Er[t-1] = math.sqrt(sqeuclidean(grad(x_prev)-Pred[t])/sqeuclidean(grad(x_prev)))

        #performance
        if t>= start:
            loss += f(x_prev)
            loss_vec = np.append(loss_vec, loss/t-start+1)
        print(t)
        
        t=t+1
    
    return loss_vec, Er, grad_serie, Pred
