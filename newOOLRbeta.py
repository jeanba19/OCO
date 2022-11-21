from scipy.optimize import minimize, Bounds
import numpy as np
import math

#in this algorithm, the predictions are made apart on the demand, pricing and contribution traces.
def simulationBeta(start,a,p,q,theta,T,m,capacity_constraint, V, sigma, next_point_option, beta):
    
    #useful functions
    
    def sqeuclidean(x):
        return np.dot(x,x)
    #objective function
    def f(x):
        return -V*a_t*np.log(1 + np.dot(x[:m],theta_t) + np.dot(x[m:],theta_t) ) + np.dot(x[:m],p_t) + np.dot(x[m:],q_t)
    #gradient function
    def grad(x):
        output = np.zeros(2*m)
        for l in range(m):
            output[l]  = -V*a_t*theta_t[l]/(1 + np.dot(x[:m],theta_t) + np.dot(x[m:],theta_t)) + p_t[l]
        for l in range(m):
            output[l+m] = -V*a_t*theta_t[l]/(1 + np.dot(x[:m],theta_t) + np.dot(x[m:],theta_t)) + q_t[l]
        return output
    
#    def grad_pred(x):
#        output = np.zeros(2*m)
#        for l in range(m):
#            output[l] = -V*a_pred[t]*theta_pred[t][l]/(1 + np.dot(x[:m],theta_pred[t]) + np.dot(x[m:],theta_pred[t])) + p_pred[t][l]
#        for l in range(m):
#            output[l+m] = -V*a_pred[t]*theta_pred[t][l]/(1 + np.dot(x[:m],theta_pred[t]) + np.dot(x[m:],theta_pred[t])) + q_pred[t][l]
#        return output


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
    t=0
    Pred[0] = grad(x1)

    #initialize error history
    H = np.zeros(T+1)
    H[0] = sqeuclidean(grad1-Pred[0])
    #Delta = np.zeros(T)
    Er = np.zeros(T)

    t=1

    while t<=T:

        #(t+1)~current values of the traces
        a_t = a[t]
        theta_t = theta[t]
        p_t = p[t]
        q_t = q[t]


        #prediction, beta is controllable error level
        
        if next_point_option == 'last-in':
            #we use last-in vector to evaluate our predicted gradient function
            #Pred[t] = grad_pred(x_prev)
            Pred[t] = grad(x_prev) - beta*grad(x_prev)
            
        else:
            #we use the argmin of \{r_{0:t}(x) + \sum_{s=1}^t \grad_s ^\top x\} to evaluate our predicted gradient function
            my_lb=np.zeros(2*m)
            my_ub=np.array(capacity_constraint*2)
            my_bounds=Bounds(my_lb,my_ub)
            estim=minimize(MinEstim, np.zeros(2*m), bounds=my_bounds)
            x_pred = estim.x

            Pred[t] = grad(x_pred) - beta*grad(x_pred)
            #Pred[t] = grad_pred(x_pred)



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
        
        #update gradient history
        acc_Grad[t] = acc_Grad[t-1]+grad(x_prev)
        grad_serie[t] = grad(x_prev)

        #update error history (error is acumulated), just withdraw H[t-1] if you wanna look at error per slot
        H[t] = H[t-1] + sqeuclidean(grad(x_prev)-Pred[t])    
        #relative percentage error history
        Er[t-1] = math.sqrt(sqeuclidean(grad(x_prev)-Pred[t])/sqeuclidean(grad(x_prev)))

        #performance
        if t >= start:
            loss += f(x_prev)
            loss_vec = np.append(loss_vec, loss/t-start+1)
        print(t)
        
        t=t+1
    
    return loss_vec, Er, Xprev
