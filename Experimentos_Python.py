import pandas
import numpy
import os
import subprocess

def BER(y, yh):
    u = numpy.unique(y)
    b = 0
    for cl in u:
        m = y == cl
        b += (~(y[m] == yh[m])).sum() / float(m.sum())
    return (b / float(u.shape[0])) * 100.

folderParams = '/shared/mgraffg/evodag-used-datasets/binary-C/evodag-0.10.6/' 
folderRes = '../res/evodag-0.10.6/TF/'
#folderParams = '../res/res/TDOF/'
#folderRes = '../res/res/TDOFselection2/'
folderData = '../data/'
ncores = 32
datasets = ['thyroid','banana','titanic','diabetis','breast-cancer','flare-solar','heart','ringnorm','twonorm','german','waveform','splice','image']
datasets_size = [100,100,100,100,100,100,100,100,100,100,100,20,20]
#datasets_size = [1,1,1,1,1,1,1,1,1,1,1,1,1]

os.chdir('/shared/cnsanchez/EvoDAG')
#os.chdir('/home/up/Documents/DOCTORADO/CODIGO/EvoDAG')
#os.chdir('/home/claudia/Documentos/DOCTORADO/CODIGO/EvoDAG')

columns=['dataset','error','fitness','size']
index=numpy.arange(len(datasets))
df = pandas.DataFrame(columns=columns,index=index)
for i in range(len(datasets)):
    archivo = datasets[i]
    error = 0.0
    fitness = 0.0
    size = 0.0
    for j in range(1,datasets_size[i]+1):
        print(archivo,j)
        fileDataTrain = folderData+archivo+'_train_'+str(j)+'.csv'
        fileDataTestData = folderData+archivo+'_test_data_'+str(j)+'.csv'
        fileDataTestLabels = folderData+archivo+'_test_labels_'+str(j)+'.csv'
        fileParams = folderParams+archivo+'_test_data_'+str(j)+ '.params'
        fileModel = folderRes+archivo+'_test_data_'+str(j)+'.model'
        filePredict = folderRes+archivo+'_test_data_'+str(j)+ '.predict'
        #os.system('EvoDAG-params -C -P ' +fileParams+' -u '+str(ncores)+' '+fileDataTrain)
        os.system('EvoDAG-train -P '+fileParams+' -m '+fileModel+' -u '+str(ncores)+' '+fileDataTrain)
        os.system('EvoDAG-predict -m '+fileModel+' -o '+filePredict+' -u '+str(ncores)+' '+fileDataTestData)
        p = subprocess.Popen(['EvoDAG-utils --size ' +fileModel],stdout=subprocess.PIPE,shell=True)
        #size += float(p.stdout.read().decode("utf-8")[6:-1])
        size = 0
        p = subprocess.Popen(['EvoDAG-utils --fitness ' +fileModel],stdout=subprocess.PIPE,shell=True)
        fitness += float(p.stdout.read().decode("utf-8")[16:-1])
        D1 = pandas.read_csv(filePredict,sep=",")
        D2 = pandas.read_csv(fileDataTestLabels,sep=",")
        e = BER(D2.values,D1.values)
        error += e
        os.system('echo '+archivo+' '+str(j)+':'+str(e))
    error /= datasets_size[i]
    size /= datasets_size[i]
    fitness /= datasets_size[i]
    df['dataset'][i] = archivo
    df['error'][i] = error
    df['fitness'][i] = fitness
    df['size'][i] = size
    print(df)
    df.to_csv(folderRes+'ares.csv',sep=',')
print(df)

