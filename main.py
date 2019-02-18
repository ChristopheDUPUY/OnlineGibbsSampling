import sys, os, random, pickle, getopt, numpy, pdb
from random import sample
random.seed(1000001)


def main(argv):
    name          = ''
    K             = 10
    alg           = ''
    kapp          = 0.5
    batch_size    = 100
    data_is_cut   = 0
    cut_size      = -1
    nb_docs       = 0
    V             = 0
    write_topics  = 0
    compute_perpl = 0
    lf            = '0'
    try:
      opts, args = getopt.getopt(argv,"n:K:a:D:B:c:p:T:V:w:C:l:",["name","K","i_algo","kappa","batch_size","data_is_cut","cut_size","nb_docs","vocabulary","write_topics","compute_perpl","logfile="])
    except getopt.GetoptError:
      print 'control.py -n <name> -K <K> -a <i_algo> -D <kappa> -B <batch_size> -c <data_is_cut> -p <cut_size> -T <nb_docs> -V <vocabulary> -w <write_topics> -C <compute_perpl> -l <logfile>'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'control.py -n <name> -K <K> -a <i_algo> -D <kappa> -B <batch_size> -c <data_is_cut> -p <cut_size> -T <nb_docs> -V <vocabulary> -w <write_topics> -C <compute_perpl> -l <logfile>'
         sys.exit()
      elif opt in ('-n','--name'):
         name = arg
      elif opt in ("-K","--K_in"):
         K = int(arg)
      elif opt in ("-a","--algo"):
         alg = arg
      elif opt in ("-D","--kappa"):
         kapp = float(arg)
      elif opt in ("-B","--batch_size"):
         batch_size = int(arg)
      elif opt in ("-c","--cut"):
         data_is_cut = bool(int(arg))
      elif opt in ("-p","--piece"):
         cut_size = int(arg)
      elif opt in ("-T","--nb_docs"):
         nb_docs = int(arg)
      elif opt in ("-V","--vocabulary"):
         V = int(arg)
      elif opt in ("-w","--write_topics"):
         write_topics = bool(int(arg))
      elif opt in ("-C","--compute_perpl"):
         compute_perpl = bool(int(arg))
      elif opt in ("-l","--logfile"):
         lf = arg

    k_alg = alg
    NO_ALPHA = False

    Gl = 'NO'
    Lo = 'NO'
    if 'V-OEM' in k_alg:
        Gl = 'FP'
        Lo = 'FP'
    elif 'G-OEM' in k_alg:
        Gl = 'FP'
        Lo = 'FP'
    if '++' not in k_alg:
        Lo = 'NO'

    if not k_alg in ["G-OEM", "G-OEM++", "V-OEM", "V-OEM++"]:
        raise ValueError("Unknown algorithm '{}', please choose among ['G-OEM', 'G-OEM++', 'V-OEM'"
                         ", 'V-OEM++']".format(k_alg))

    prefix  = name +'_K' +str(K)
    path    = './data/'
    pathR   = './Results/'
    os.system('mkdir -p ./Results/')

    if lf=='0':
        print '#####################'
        print '   '+name
        print '   '+k_alg
        print '   K='+str(K)
        print '   kappa=',kapp
        print '#####################'
    else:
        logfile=open(lf,'a')
        logfile.write('#####################\n   '+prefix+'\n   '+k_alg+'\n   K='+str(K)+'\n   kappa='+str(kapp)+'\n#####################\n')
        logfile.close()


    if (not os.path.isfile(path+name+'_ids.pck') and not data_is_cut) or (data_is_cut and not os.path.isfile(path+name+'_ids_B0.pck')):
        if lf=='0':
            print '      Convert data...'
        else:
            logfile=open(lf,'a')
            logfile.write('      Convert data...\n')
            logfile.close()
        execfile('./convert.py')

    execfile('./control.py')

    if lf=='0':
        print 'DONE!'
    else:
        logfile=open(lf,'a')
        logfile.write('DONE!')
        logfile.close()

if __name__== "__main__":
   main(sys.argv[1:])

