import pandas
import numpy
import os
import subprocess
import gzip
import pickle

def BER(y, yh):
    u = numpy.unique(y)
    b = 0
    for cl in u:
        m = y == cl
        b += (~(y[m] == yh[m])).sum() / float(m.sum())
    return (b / float(u.shape[0])) * 100.

res = ['EvoDAG','TF','TDOF','TF_params','TDOF_params']
folderRes = ['/shared/mgraffg/evodag-used-datasets/binary-C/evodag-0.10.6/','../res/evodag-0.10.6/TF/','../res/evodag-0.10.6/TDOF/','../res/evodag-0.10.6/TF_params/','../res/evodag-0.10.6/TDOF_params/']
folderData = '../data/'
datasets = ['thyroid','banana','titanic','diabetis','breast-cancer','flare-solar','heart','ringnorm','twonorm','german','waveform','splice','image']
datasets_size = [100,100,100,100,100,100,100,100,100,100,100,20,20]
#datasets_size = [1,1,1,1,1,1,1,1,1,1,1,1,1]

os.chdir('/shared/cnsanchez/EvoDAG')
#os.chdir('/home/up/Documents/DOCTORADO/CODIGO/EvoDAG')

####### BER Table ###########################################
strv = 'Data set '
for i in range(len(res)):
    strv += ' & '+res[i]
print('\hline')
print(strv + '\\\\')
print('\hline')

for i in range(len(datasets)):
    archivo = datasets[i]
    
    strv = archivo
    for k in range(len(res)):
        ber = []
        for j in range(1,datasets_size[i]+1):
            fileDataTestLabels = folderData+archivo+'_test_labels_'+str(j)+'.csv'
            filePredict = folderRes[k]+archivo+'_test_data_'+str(j)+ '.predict'
            D1 = pandas.read_csv(filePredict,sep=",")
            D2 = pandas.read_csv(fileDataTestLabels,sep=",")
            ber.append(BER(D2.values,D1.values))
        v = numpy.median( numpy.array(ber) )
        strv += '& '+str( round(v,4))
    print(strv + '\\\\ \\hline')
print(' ')

####### Fitness Table ###########################################
strv = 'Data set '
for i in range(len(res)):
    strv += ' & '+res[i]
print('\hline')
print(strv + '\\\\')
print('\hline')

for i in range(len(datasets)):
    archivo = datasets[i]
    
    strv = archivo
    for k in range(len(res)):
        value = []
        for j in range(1,datasets_size[i]+1):
            fileModel = folderRes[k]+archivo+'_test_data_'+str(j)+'.model'
            with gzip.open(fileModel, 'r') as fpt:
                m = pickle.load(fpt)
                value.append( m.fitness_vs )
                #print("Height: %s" % m.height)
                #print("Size: %s" % m.size)
                #print("Fitness vs: %s" % m.fitness_vs)
        v = numpy.median( numpy.array(value) )
        strv += '& '+str( round(v,4))
    print(strv + '\\\\ \\hline')
print(' ')

####### Size Table ###########################################
# strv = 'Data set '
# for i in range(len(res)):
#     strv += ' & '+res[i]
# print('\hline')
# print(strv + '\\\\')
# print('\hline')
# for i in range(len(datasets)):
#     archivo = datasets[i]
#     strv = archivo
#     for k in range(len(res)):
#         value = []
#         for j in range(1,datasets_size[i]+1):
#             fileModel = folderRes[k]+archivo+'_test_data_'+str(j)+'.model'
#             with gzip.open(fileModel, 'r') as fpt:
#                 m = pickle.load(fpt)
#                 value.append( m.size )
#                 #print("Height: %s" % m.height)
#                 #print("Size: %s" % m.size)
#                 #print("Fitness vs: %s" % m.fitness_vs)
#         v = numpy.median( numpy.array(value) )
#         strv += '& '+str( round(v,4))
#     print(strv + '\\\\ \\hline')