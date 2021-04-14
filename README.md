## EACoupledCF
This is about the code' some introduce information  of the  EACoupledCF

## Citation
'''
@inproceedings{
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
}
'''

## Dataset
We provide two processed datasets: ML1M and Tafeng
ML1M:
* ml-1m.train.rating 
  * train file.
  * eacg line is a user with one of her/his iteracted items and  ratings and timesatmp(useless)
* ml-1m.test.rating 
  * test file.
  *same format with the train file
* ml-1m.test.negative 
  * test negative file. each row contains a user and one postive item  and ninty-nine negative items
  * the format is : (u,pos_i) neg_i 
  
 #movies.dat
 
 #rating.dat
 
 #u.age
 
 #u.genre
 
 #u.info
 
 #u.occpation
 
 #users.dat
 

Tafeng:
  *test.negative
  *test.rating
  *train.rating
  *item.data
  *rating.data
  *user.data
 
 
## Enviroment
  python 3.6.12
  keras 2.1.4
  tensorflow-gpu 1.5.0
  numpy 1.19.2

 


