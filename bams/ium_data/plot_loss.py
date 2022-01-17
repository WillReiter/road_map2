import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(path, epoch_losses, epoch):  
    '''
    input_dtype : list 
    '''
    
    x , y = [], []
    for i,value in enumerate(epoch_losses):
        x.append((i+1)*epoch)
        y.append(round(value, 4))
    plt.plot(x,y, color=(0,0,1), label='loss') 
    min_index = (np.argmin(epoch_losses)+1)*epoch  ## np.argmin()
    min_value = round(np.min(epoch_losses), 4)
    plt.plot(min_index, min_value, "o")
    # plt.plot(min_index, min_value, "ro")
    show_min = "[{0}, {1}]".format(min_index, min_value)
    
    #plt.xticks(range(epoch, (len(epoch_losses)+1)*epoch, epoch))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train_Loss")

    plt.annotate(show_min, xytext=(-20,10),textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5',\
                  fc='yellow', ec='k', lw=1, alpha=0.5),xy=(min_index, min_value))
    #plt.show()
    plt.savefig(os.path.join(path, 'train_loss.jpg'))
    plt.close()

if __name__ == '__main__':
    a = {5000	:1.2795,
         10000	:0.7883,
         15000	:0.6983,
         20000	:0.6329,
	 25000	:0.58,
	 30000	:0.5363,
	 35000	:0.477,
	 40000	:0.4548,
	 45000	:0.446,
	 50000	:0.4344,
	 55000	:0.4238,
	 60000	:0.4177,
	 65000	:0.4107,
	 70000	:0.4026,
	 75000	:0.4023,
	 80000	:0.4029,
	 85000	:0.4066,
	 90000	:0.4007,
	 95000	:0.4006,
	 100000	:0.4017,
	 105000	:0.3983,
	 110000	:0.3992,
	 115000	:0.404,
	 120000	:0.3982,
	 125000	:0.4035,
	 130000	:0.3946,
	 135000	:0.3994,
	 140000	:0.3954,
	 145000	:0.3953,
	 150000	:0.3944}
    #a = [16.19429655075073, 14.266041564941407, 8.14479832649231, 5.792479038238525, 4.923104596138001]
    a = list(a.values())
    plot_loss('./', a, 5000)
