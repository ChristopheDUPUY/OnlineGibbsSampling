# -*- coding: utf-8 -*-
import numpy,pickle,time,scipy
import warnings
warnings.simplefilter("error", RuntimeWarning)
import onlineLDA
from numpy import log,exp,sqrt
from scipy.special import gamma,polygamma

numpy.random.seed(200)

# Test set
if compute_perpl:
    fname_test  = path+name+'_ids_test.pck'
    ids_test   = pickle.load(open(fname_test,'r'))


if lf=='0':
    print '   ---->',name
    print '   ---->',nb_docs,'observations'
    print '   ---->',K,'guesses'
    print '   ---->',batch_size,'elements/batch'
else:
    logfile=open(lf,'a')
    logfile.write('   ---->'+name+'\n'+'   ---->'+str(nb_docs)+'observations'+'\n'+'   ---->'+str(K)+'guesses\n   ---->'+str(batch_size)+'elements/batch\n')
    logfile.close()
    


start      = time.time()
# Build model
model = onlineLDA.OnlineEM_LDA(V,batch_size,K,nb_docs,1.0,nb_GS=20,kappa=kapp,Gl=Gl,Lo=Lo)
prefix += '_Gl'+Gl+'_Lo'+Lo

# Main loop over the minibatches
if data_is_cut:
    fname_train = path+name+'_ids_B0.pck'
    for i_b in range(0,nb_docs,cut_size):
        temp_name = fname_train.replace('B0','B'+str(i_b))
        ids_b = pickle.load(open(temp_name,'r'))
        for i_batch in range(0,min([cut_size,nb_docs-i_b]),batch_size):
            if lf=='0':
                print '             i_batch:'+str(i_b+i_batch)+'/'+str(nb_docs)+' - '+str(round(time.time() - start,1))
            else:
                logfile=open(lf,'a')
                logfile.write('             i_batch:'+str(i_b+i_batch)+'/'+str(nb_docs)+' - '+str(round(time.time() - start,1))+'\n')
                logfile.close()
            model.Key_Step(k_alg,ids_b[i_batch:i_batch+batch_size])
else:
    fname_train = path+name+'_ids.pck'
    ids_b = pickle.load(open(fname_train,'r'))
    for i_batch in range(0,nb_docs,batch_size):
        if lf=='0':
            print '             i_batch:'+str(i_batch)+'/'+str(nb_docs)+' - '+str(round(time.time() - start,1))
        else:
            logfile=open(lf,'a')
            logfile.write('             i_batch:'+str(i_batch)+'/'+str(nb_docs)+' - '+str(round(time.time() - start,1))+'\n')
            logfile.close()

        model.Key_Step(k_alg,ids_b[i_batch:i_batch+batch_size])

if lf=='0':
    print '             i_batch:'+str(nb_docs)+'/'+str(nb_docs)+' - '+str(round(time.time() - start,1))
else:
    logfile=open(lf,'a')
    logfile.write('             i_batch:'+str(nb_docs)+'/'+str(nb_docs)+' - '+str(round(time.time() - start,1))+'\n')
    logfile.close()

f0 = open(pathR+prefix+'_kapp'+str(kapp)+'_eta1_'+k_alg+'.pck','w');pickle.dump(model.eta1,f0);f0.close()
f0 = open(pathR+prefix+'_kapp'+str(kapp)+'_eta2_'+k_alg+'.pck','w');pickle.dump(model.eta2,f0);f0.close()


if write_topics:
    if lf=='0':
        print 'WRITE TOPICS - ',round(time.time() - start,1)
    else:
        logfile=open(lf,'a')
        logfile.write('WRITE TOPICS - '+str(round(time.time() - start,1))+'\n')
        logfile.close()
    vocab = open(path+name+'_vocab.txt','r').readlines()
    onlineLDA.ExtractTopics(K,vocab,model.eta1,model.eta2,pathR+prefix+'_kapp'+str(kapp)+'topics.txt',words_max=20)
    


if compute_perpl:
    if lf=='0':
        print 'PERPLEXITY - ',round(time.time() - start,1)
    else:
        logfile=open(lf,'a')
        logfile.write('PERPLEXITY - '+str(round(time.time() - start,1))+'\n')
        logfile.close()

    perf_g   = model.Perplexity_test(ids_test,model.eta1,model.eta2,lf=lf)
    f0 = open(pathR+prefix+'_kapp'+str(kapp)+'_perf_'+k_alg+'.pck','w');pickle.dump(perf_g,f0);f0.close()

