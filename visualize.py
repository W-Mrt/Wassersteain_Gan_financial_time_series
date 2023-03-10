import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def loss_history(date):
    def subsample(x): #take the samples from one epoch
        indices = [i for i in range(0,1000)]
        return x[indices]
    g_loss = np.load('./imgs/%s/g_loss.npy'%(date))
    d_loss = np.load('./imgs/%s/d_loss.npy'%(date))
    plt.figure(figsize=(24,15),dpi=800)
    ax = plt.subplot(100)
    plt.xlabel('batches',fontsize=36)
    plt.ylabel('loss',fontsize=36)
    indices = [i for i in range(0,1000)]
    ax.plot(indices,subsample(g_loss),'-',linewidth=4,color='red',alpha=0.8)
    ax.plot(indices,subsample(d_loss),'-',linewidth=4,color='blue',alpha=0.8)
    plt.yscale('log')
    plt.ylim(0.01,14)
    plt.xlim(0,1000)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    box = ax.get_position()
    plt.savefig('%s_losses'%date,transparent=True)
    plt.close()

def time_series(x,file_name):
    plt.figure(dpi=150)
    plt.plot(x)
    plt.ylim(-0.25,0.25)
    plt.xlabel('Time Ticks($t$)',fontsize=20)
    plt.ylabel('Normalized Return',fontsize=20)
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()

def leverage_effect(x,y,file_name):          
    plt.figure(dpi=150)
    plt.plot(x,y)
    plt.xlabel(r'$t$',fontsize=20)
    plt.ylabel(r'$L(t)$',fontsize=20)
    plt.axhline()
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()

def distribution(x,y,file_name,scale='linear'):
    if scale is 'log':
        distribution_log(x, y, file_name)
    elif scale is 'linear':
        distribution_linear(x, y, file_name)

def distribution_linear(x,y,file_name):
    plt.figure(dpi=150)
    plt.plot(x,y,'.',markersize=8)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Normalized Scale in $\sigma$',fontsize=20)
    plt.ylabel('PDF $P(r)$',fontsize=40)
    plt.savefig(file_name+'_linear.png',transparent=True)
    plt.close()


def distribution_log(x,y,file_name):
    #split into positive and negative sides
    dist_x_pos = x[x > 0]
    dist_y_pos = y[-dist_x_pos.size:]
    dist_x_neg = -x[x < 0]
    dist_y_neg = y[:dist_x_neg.size]
    #for positive
    plt.figure(dpi=150)
    plt.plot(dist_x_pos,dist_y_pos,'.')
    plt.xlabel('Normalized Scale in $\sigma$',fontsize=20)
    plt.ylabel('PDF $P(r)$',fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(file_name+'_pos_log.png', transparent=True)
    plt.close()
    #for negative
    plt.figure(dpi=150)
    plt.plot(dist_x_neg,dist_y_neg,'.')
    plt.xlabel('Normalized Scale in $\sigma$',fontsize=20)
    plt.ylabel('PDF $P(r)$',fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(file_name+'_neg_log.png', transparent=True)
    plt.close()


def acf(acf_values,file_name,scale='log',):
    plt.figure(dpi=150)
    plt.plot(np.linspace(1,acf_values.size,acf_values.size),acf_values,'.')
    plt.ylim(1e-5,1.)
    if scale is 'linear':
        plt.ylim(-1.,1.)
    plt.xscale('log')
    plt.yscale(scale)
    plt.xlabel('lag $k$',fontsize=20)
    plt.ylabel('Auto-correlation',fontsize=20)
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()