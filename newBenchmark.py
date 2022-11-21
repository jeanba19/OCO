import numpy as np
import math
import ipopt



def benchmark(start,a,p,q,theta,T,m,capacity_constraint,V):
    
    #create class of benchmark solution
    class probleme(object):
        def __init__(self):
            pass

        def objective(self, x):

            output = 0

            for tprim in range(start,t+1):
                output += -V*a[tprim]*np.log(1 + np.dot(x[:m],theta[tprim]) + np.dot(x[m:],theta[tprim])) + np.dot(x[:m],p[tprim]) + np.dot(x[m:],q[tprim])

            return output
            #The callback for calculating the objective

        def gradient(self, x):

            output = np.zeros(2*m)

            for tprim in range(start,t+1):

                for i in range(m):
                    output[i] += -V*a[tprim]*theta[tprim][i]/(1 + np.dot(x[:m],theta[tprim]) + np.dot(x[m:],theta[tprim])) + p[tprim][i]
                for i in range(m):
                    output[m+i] += -V*a[tprim]*theta[tprim][i]/(1 + np.dot(x[:m],theta[tprim]) + np.dot(x[m:],theta[tprim])) + q[tprim][i]

            return output
        
        
    #create static benchmark instance

    lb = np.zeros(2*m).tolist()
    ub = capacity_constraint*2

    x00 = np.zeros(2*m).tolist()

    nlp = ipopt.problem(n=len(x00),
                        m=0,
                        problem_obj=probleme(),
                        lb=lb,
                        ub=ub,
                        cl=None,
                        cu=None)
    
    t=start
    #benchmark performance
    opt_loss = 0
    opt_loss_vec = np.array([])
    Xopt = np.zeros((T+1,2*m))

    while t <=T:

        #current slot information
        #a_t = a[t]
        #theta_t = theta[t]
        #p_t = p[t]
        #q_t = q[t]

        #benchmark decision (ipopt non-linear solver)
        xopt, info = nlp.solve(x00)
        
        opt_loss = info['obj_val']
        Xopt[t] = xopt
        opt_loss_vec=np.append(opt_loss_vec,opt_loss/t-start+1)
        print(t)
        t+=1
        
    return opt_loss_vec, Xopt
