# BiDens

The source code is developed based on Python3.
The hash function that we use is murmurhash3, so the Python lib mmh3 should be installed before running.
We compare our BiDens with densification methods OPTDens, FastDens, and a method combining both OPTDens and FastDens (referred to as OPT-FastDens).
      
similarity_synthetic.py is develop for estimating the similarity between two synthetic sets. It contains four parameters: k, sketch size; jcd, the predefined Jaccard similarity between these two sets; rounds, the number of repeated experiments; cardinality, the number of distinct elements in each set.

runtime_synthetic.py is developed for measuring the runtime during which we implement each densification method to build a densified sketch. It contains three parameters: k, sketch size; gamma, the ratio of empty bins in the sketch; rounds, the number of repeated experiments.

The folder "parameters" contains the parameters used by our method BiDens, and we list four files that consist of the parameters when we set the sketch size as k=100.
