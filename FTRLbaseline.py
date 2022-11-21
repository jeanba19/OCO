from scipy.optimize import minimize, Bounds
import numpy as np
import math


def Forel(start,a,p,q,theta,T,m,capacity_constraint, V, eta):
    
    #useful functions
    
    
   

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
        return (1/(2*eta)) * sqeuclidean(x-Xprev[t-1])



    def acc_grad(x):

        return np.dot(acc_Grad[t-1],x)

    # time slot t will be defined as a global variable, not an argument    
    def toMinimize(x):
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

    

    t=1

    while t<=T:     



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
        
        #(t+1)~current values of the traces
        a_t = a[t]
        theta_t = theta[t]
        p_t = p[t]
        q_t = q[t]
        
        
        
        #update gradient history
        acc_Grad[t] = acc_Grad[t-1]+grad(x_prev)
        grad_serie[t] = grad(x_prev)
        




        #performance
        if t>= start:
            loss += f(x_prev)
            loss_vec = np.append(loss_vec, loss/t-start+1)
        print(t)
        
        t=t+1
    
    return loss_vec, grad_serie
