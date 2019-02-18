Online but Accurate Inference for Latent Variable Models with Local Gibbs Sampling
Christophe Dupuy and Francis Bach, 2017

Contact:
--------
Christophe Dupuy
Francis Bach

Citation:
---------
Online but Accurate Inference for Latent Variable Models with Local Gibbs Sampling
Christophe Dupuy and Francis Bach, JMLR 2017
http://www.jmlr.org/papers/volume18/16-374/16-374.pdf
https://hal.inria.fr/hal-01284900



How to use:
-----------

python2.7 main.py -n <name> -K <K> -a <algo> -D <kappa> -B <batch_size> -c <data_is_cut> -p <cut_size> -T <nb_docs> -V <vocabulary> -w <write_topcis> -C <compute_perpl> -l <logfile>

     - <name>         : name of the database,
     - <K>            : number of topics inferred,
     - <algo>         : algorithm (G-OEM is our algorithm, see our paper for other notations: G-OEM++, V-OEM, V-OEM++),
     - <kappa>        : stepsize exponent,
     - <batch_size>   : size of the minibatch,
     - <data_is_cut>  : 0 if there is only one data file, 1 if the data is cut in multiple files,
     - <cut_size>     : number of documents per file,
     - <nb_docs>      : total number of documents,
     - <vocabulary>   : number of words in the vocabulary,
     - <write_topcis> : if 1, the topics inferred are saved in a file in ./Results/,
     - <compute_perpl>: if 1, the perplexity is computed at the end of the process,
     - <logfile>      : (optional) path to the log.


The data files must be formatted as follows:
   - vocab.txt            : each line is a word of the vocabulary
   - name+'_ids_B0.txt'   : Part of the dataset (this format corresponds to data_is_cut=1). Each line of this file is document, represented as integers separated by a space; 'n1 n2 n3 n4...'. See data/ for examples.
   - name+'_ids.txt'      : Entire dataset (this format corresponds to data_is_cut=0). Same representation as above.
   - name+'_ids_test.txt' : Test set of documents for evaluation. Same representation as above.



Example of use:
---------------
Run G-OEM, 100 topics inferred on a subset of New York Times dataset from UCI (https://archive.ics.uci.edu/ml/datasets/Bag+of+Words)

python2.7 main.py -n NYT -K 100 -a G-OEM -D 0.5 -B 100 -c 1 -p 10000 -T 20000 -V 44228 -w 1-C 0
or 
python2.7 main.py -n NYT -K 100 -a G-OEM -D 0.5 -B 100 -c 1 -p 10000 -T 20000 -V 44228 -w 1 -C 0 -l ./test.log

