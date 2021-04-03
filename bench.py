import numpy as np

def benchmark(T,D,B,K,A,Q,P):
    
    import ipopt
    
    class probleme(object):
        def __init__(self):
            pass

        def objective(self, x):
            output=0
            for k in range(K):
                output += -a_k[k]*np.log(x[0]+x[k+1]+1)
            return output
            #The callback for calculating the objective


        def gradient(self, x):
            # The callback for calculating the gradient
            output=np.zeros(K+1)
            for k in range(K):
                output[0] += -a_k[k]/(1+x[0]+x[k+1])
            for k in range(1,K+1):
                output[k] = -a_k[k-1]/(1+x[0]+x[k])
            return output



        def constraints(self, x):
            #
            # The callback for calculating the constraints
            #
            return np.dot(mult_t,x)

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            return mult_t
    
    A_k_it = A[:K*T+K]
    A_k_it = A_k_it.reshape(T+1,K)
    
    Q_k_it = Q[:K*T+K]
    Q_k_it = Q_k_it.reshape(T+1,K)
    
    new_column = P
    Mult_t = np.insert(Q_k_it, 0, new_column, axis=1)
    
    
    x0 = np.ones(K+1).tolist()

    lb = np.zeros(K+1).tolist()
    ub = (np.ones(K+1)*D).tolist()

    cl = [0.0]
    cu = [B]


    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=probleme(),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )
    
    t=0
    opt_revenue=0
    opt_results=np.array([])
    opt_fit=0
    opt_vect_fit=np.array([])
    Xopt=np.empty((T+1,K+1))
    U_z_T = 0
    while t <=T:
        #udpdate demand and price
        a_k = A_k_it[t]
        mult_t = Mult_t[t]
        #benchmark
        x, info = nlp.solve(x0)
        Xopt[t]=x
        
        if t>=1:
            U_z_T += np.linalg.norm(Xopt[t-1]-Xopt[t])
   
            opt_revenue += info['obj_val']
            opt_results=np.append(opt_results,opt_revenue/t)
        
        

            opt_fit += (info['g']-B)
            opt_vect_fit=np.append(opt_vect_fit, opt_fit/t)
        
        t=t+1
        
        
    return opt_results, opt_vect_fit, U_z_T, Xopt