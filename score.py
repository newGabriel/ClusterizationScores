import numpy as np


'''purity_score(y,y_pre)

    Computes the purity score.
    
    @arg y: np_array ground truth
    
    @arg y_pre: result of the sklearn clusterization
    
    @return: value of the clusterization's purity
    
'''
def purity_score(y,y_pre):
    c = list()
    for i in range(max(y_pre)+1):
        c.append(list())
    
    for i in range(len(y_pre)):
        c[y_pre[i]].append(i)
    
    t = 0
    for i in c:
        p = np.zeros(y.max())
        for j in i:
            p[y[j]-1] += 1
        
        t += p.max()
    
    pr = t/float(len(y))
    return pr
   
 
'''collocation_score(y,y_pre)

    Computes the colocation score.
    
    @arg y: np_array ground truth
    
    @arg y_pre: result of the sklearn clusterization
    
    @return: value of the clusterization's colocation
    
'''
def collocation_score(y,y_pre):
    c = list()
    for i in range(max(y_pre)+1):
        c.append(list())
    
    for i in range(len(y_pre)):
        c[y_pre[i]].append(i)
        
    t = 0
    y1 = np.array(range(len(y)))
    y1 = y1.reshape((41,int(len(y)/41)))
    for i in y1:
        p = np.zeros(len(c))
        for j in i:
            for k in range(len(c)):
                if j in c[k]:
                    p[k] += 1
        t += p.max()
    pr = t/float(len(y))
    return pr
    

'''harmonicMean_score(y=0,y_pre=0,p=0,c=0)

    Computes the Harmonic Mean score.
    
    @arg y: np_array ground truth
    
    @arg y_pre: result of the sklearn clusterization
    
    or
    
    @arg p: pre computed purity value
    
    @arg c: pre computed collocation value
    
    @return: value of the clusterization's harmonic mean
    
'''
def harmonicMean_score(y=0,y_pre=0,p=0,c=0):
    if p==c==0:
        p = purity_score(y,y_pre)
        c = collocation_score(y,y_pre)
    return (2*c*p)/(c+p)

