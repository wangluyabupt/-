import numpy as np
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

Background = [0,0,0]
DaoGuan = [255, 255, 255]
Else = [150, 150, 150]
YouZhuganjin = [255, 128, 0]
YouZhuganzhong = [255, 0, 128]
YouZhuganyuan = [0, 255, 128]
HouJiangzhiyou = [128, 128, 255]
HouCezhi = [(128, 255, 128)]

DSET_MEAN = [0.45028183821135875, 0.45028183821135875, 0.45028183821135875]
DSET_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]
#RESULTS_PATH = '/home/yrl/100layers/tiramisu/tiramisu_vessels/result/'
RESULTS_PATH = '/home/wly/Documents/100layers_bi/tiramisu/tiramisu_vessels/result/'
label_colours = np.array([Background, DaoGuan, Else, YouZhuganjin,
           YouZhuganzhong, YouZhuganyuan, HouJiangzhiyou, HouCezhi])

def view_annotated(tensor, plot=False):
    temp = tensor.numpy()
    print ('temp:',temp,temp.shape)
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()    
    for l in range(0,0):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    #plt.imshow(rgb)
    #plt.axis('off')
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
    #plt.margins(0,0)
    #plt.savefig('/home/yrl/100layers/tiramisu/tiramisu_vessels/result/31_a_b_27_1_src_LAO_0006.png')
    #ave_pic = Image.fromarray(rgb)##########
    #ave_pic.save( RESULTS_PATH + '31_a_b_27_1_src_LAO_0006.png')##########
    #if plot:
        #plt.imshow(rgb)
        #plt.show()
    #else:
    return rgb

def decode_image(tensor):
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array(DSET_MEAN)
    std = np.array(DSET_STD)
    inp = std * inp + mean
    return inp

def view_image(tensor):
    inp = decode_image(tensor)
    inp = np.clip(inp, 0, 1)
    ##plt.imshow(inp)
    #plt.show()
