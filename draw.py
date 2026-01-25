import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

folder = './result/cifar'
# folder = './iid_dirichlet_0.1'
# folder = './iid_dirichlet_1'
# folder = './iid_mixed_0.95'
data_filenames=[
    '/acc_cifar10_cnn_clsrand_R[1000]_l_[1]_acc[0.4266]_time[16064.944].npy',
    '/acc_cifar10_cnn_rand_R[1000]_l_[1]_acc[0.4941]_time[18412.415].npy',
    '/acc_cifar10_cnn_randcls_R[1000]_l_[1]_acc[0.4826]_time[14933.645].npy'
    ]

def mv_avg(a,window_size,mode="valid"):
    return(np.convolve(a, np.ones(window_size)/window_size, mode=mode))

def mv_std(x,window_size):
    stds=np.zeros_like(x)
    for i in range(window_size,len(x)):
        stds[i]=np.std(x[i-window_size:i])
    return np.array(stds)
    

window_sz=10

plt.rcParams['pdf.fonttype'] = 42
f=plt.figure(figsize=(16,12))
sns.set_context(rc={"lines.linewidth": 2.5})
sns.set_palette(palette='muted')
p=[]
for file in data_filenames:
    path = folder + file
    data = np.load(path, allow_pickle = True)
    data = data[:1000]

    # std=mv_std(acc,window_sz)
    # std=std[:-window_sz+1]
    
    x = list(range(len(data)-window_sz+1))
    smooth_acc = mv_avg(data,window_sz,'valid')
    std = np.std(smooth_acc[-50:])

    fig, = plt.plot(smooth_acc)
    p.append(fig)
    plt.fill_between(x,smooth_acc+std,smooth_acc-std,alpha=0.3)
    # plt.fill_between(x,smooth_acc[:-window_sz+1]+std/2, smooth_acc[:-window_sz+1]-std/2,alpha=0.3)
    
plt.grid(True)
plt.xlim([0,len(x)])
# plt.ylim([40,85])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylabel('Test accuracy',fontsize=37)
plt.xlabel('Communication rounds',fontsize=37)

plt.legend(p,['rand-cluster', 'fedavg', 'sequence-cluster']
# plt.legend(p,['FedAvg','FedLP_Homo(0.7)','FedLP_Homo(0.5)','FedLP_Homo(0.3)','FedLP_Hetero(u)','FedLP_Hetero(5)','FedLP_Hetero(3)','FedLP_Hetero(1)']
# plt.legend(p,['FedAvg','FedLP_Homo(0.7)','FedLP_Homo(0.5)','FedLP_Homo(0.3)','FedLP_Hetero(u)','FedLP_Hetero(5)']
                ,fontsize=35,loc='lower right')
# plt.show()
f.savefig(folder+'_acc_3.pdf')