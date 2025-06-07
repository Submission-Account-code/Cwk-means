                    Readme

This folder contains 22 data files used in our paper "Influential Features PCA for High dimensional Clustering"
and 1 folder. The folder contains relevant papers referred in the instroction of data files. 


The data files are:

BhattacherjeeLung2001.txt: data matrix based on 203 lung tumors/normal lung samples 
on mRNA expression levels of  12,600 transcripts, which is for analysis on different 
subclasses  of lung adenocarcinoma. The first 139 columns correspond 
to lung tumors with adenocarcinomas, and the remaining 64 columns 
correspond to other lung tumors and normal lung samples.  

BhattacherjeeLung2001.y.txt: 203 by 1 label matrix. Label 0 indicates 
lung tumors with adenocarcinomas, and label 1 indicates other lung tumors 
and normal lung samples.


Relevant papers: Bhattacherjee et al. (2001), Yousefi (2009), Bioinformatics.




brain.x.txt, data matrix based on Brain A data in Pomeroy et al 
(2002) and pre-processed by Dettling. It contains 5597 genes on 42 samples.

brain.y.txt: vector of class index. This is a multi-class problem (5 classes in total).

Related papers: Dettling Bioinformatics (2004), Pomeroy et al. Nature 
(2002). 



colon.x.txt, data matrix based on Colon data in Alon et al. (1999). 
The original data is 2000 by 62, ready for use in Matlab.

colon.y.txt, vector of class index (62 by 1).

Related papers: Dettling Bioinformatics (2004), Alon et al. PNAS
(1999).



Leukemia.mat: original data matrix based on 73 samples on Leukemia measured 
on the same set of 7129 genes from Golub et al. (1999). Used for normalizaing 
SuCancer data set.

leukemia.x.txt: data matrix pre-processed by Dettling (2004), with only 
72 samples and 3571 genes left. Ready for use in Matlab.   

leukemia.y.txt: vector of class index (72 by 1). 

Relevant papers: Golub et al. (1999). Dettling (2004), Bioinformatics.


lungcancer.txt: data matrix based on the Lung Cancer microarray data measured on 
12,533 genes on 181 samples from Gordon et al (2002).  

lungcancery.txt: vector of class index (181 by 1).

Relevant papers: Gordon et al (2002).




lymphoma.x.txt: data matrix based on Lymphoma data in Alizadeh et al. (2000). 
The 4026 by 62 data matrix is ready for use in Matlab.

lymphoma.y.txt: vector of class index (62 by 1). This is a multi-class 
classification problem (3 classes).

Relevant papers: Dettling Bioinformatics (2004), Alizadeh et al. (2000), Nature.



prostate.x.txt: data matrix based on Prostate data set used in Welsh et al. (2001) 
and Singh et al. (2002). With pre-processing in Dettling (2004), only 6033 genes are left.   
prostate.y.txt: vector of class index (102 by 1). 

Relevant paper: Singh et al. (2002), Welsh et al. (2001), Fan (2007), and Dettling (2004). 



srbct.x.txt: data matrix based on SRBCT data in Kahn et al. (2001). This 2308 by 63
data matrix is ready for use in Matlab.

srbct.y.txt: vector of class index (63 by 1). This is a multi-class classification 
problem (4 classes).

Related papers: Dettling Bioinformatics (2004), Kahn et al. (2001), Nat. Med.



SuCancer2001.txt: data matrix containing results from 174 samples on different tumor cells 
measured on the same set of 12,533 transcripts from Su et al. (2001). The first 83 columns 
correspond to tumors with bladder/ureter, breast, colorectal and prostate, the remaining 91 
columns correspond to other 7 types of cancers. 

SuCancer.txt: processed data matrix. In this matrix, only 7909 features are 
kept. The first 83 columns correspond to tumors with bladder/ureter, breast, 
colorectal and prostate, the remaining 91 columns correspond to other 7 types 
of cancers. The cleaning method is the same with that in Dettling and Buhlmann(2003).

SuCancer.y.txt: 174 by 1 label matrix. Label 0 indicates samples with bladder/ureter, 
breast, colorectal and prostate, andlabel 1 indicates samples with other 7 types of cancers. 
The label is according to results in Yousefi (2009).


Relevant papers: Su et al. (2001), Yousefi (2009), Dettling and Buhlmann(2003), Bioinformatics.





WangBreast2005.txt: data matrix based on 276 samples on Lymph-node-negative 
breast cancer measured on the same set of 22,215 transcripts from Wang et al. (2005), which 
is to identify genes that indicated whether patients will develop distant metastases within 
5 years or not. The first 183 columns correspond to patients had metastases or follow-up 
larger than 5 years, the remaining 93 columns correspond to patients had metastases within 
5 years. 

WangBreast2005.y.txt: 276 by 1 label matrix. Label 0 indicates patients 
had metastases or follow-up larger than 5 years, and label 1 indicates 
patients had metastases within 5 years. 
The label is according to results in Yousefi (2009).


Relevant papers: Wang et al. (2005), Yousefi (2009), Bioinformatics.