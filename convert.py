import os,pickle

if data_is_cut:
    for i_b in range(0,nb_docs,cut_size):
        if lf=='0':
            print '         Part: '+str(i_b)
        else:
            logfile=open(lf,'a')
            logfile.write('         Part: '+str(i_b)+'\n')
            logfile.close()
        data = open(path+name+'_ids_B%s.txt'%(str(i_b)),'r').readlines()
        ids  = [[int(i) for i in r.split(' ')] for r in data]

        f0 = open(path+name+'_ids_B%s.pck'%(str(i_b)),'w')
        pickle.dump(ids,f0)
        f0.close()

else:
    data = open(path+name+'_ids.txt','r').readlines()
    ids  = [[int(i) for i in r.split(' ')] for r in data]

    f0 = open(path+name+'_ids.pck','w')
    pickle.dump(ids,f0)
    f0.close()

if lf=='0':
    print '         Test'
else:
    logfile=open(lf,'a')
    logfile.write('         Test\n')
    logfile.close()

data = open(path+name+'_ids_test.txt','r').readlines()
ids  = [[int(i) for i in r.split(' ')] for r in data]

f0 = open(path+name+'_ids_test.pck','w')
pickle.dump(ids,f0)
f0.close()
