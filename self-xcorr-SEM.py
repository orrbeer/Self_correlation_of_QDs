import matplotlib.pyplot as plt
#############################################################################
SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"
##############################################################################
import numpy as np
from skimage import data
from skimage.feature import match_template
from skimage import io
from skimage.color import rgb2gray
from scipy.fftpack import fftn, fftshift
from skimage.filters import difference_of_gaussians, window
from scipy.ndimage import gaussian_filter
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH_RECIPROCAL
from matplotlib.colors import LogNorm

images = ['50nM 1000rpm_01 (1).tif', 'star03.tif']
fig = plt.figure(figsize=(12, 5))
for i in range(2):
    image = io.imread(r'C:\Users\orrbeer\Desktop\PhD\Reports\super paper\Raw data sample for Technion\SEM\{}'.format(images[i]))#, img_num=10)
    image = image[:1420,:]

    # print(np.amax(image))
    # fig = plt.figure(figsize=(8, 8))
    # ax2 = plt.subplot(111)
    # ax2.imshow(image)
    
    if i == 0:
        scalebar =  ScaleBar( 0.925925926, 'nm') #  321-254=108 pixels are 100 nm
        ft_scalebar = ScaleBar( 0.000527616141, 'nm') # total pixels 2047 image length = 2047*0.9259 fftpixel = 1/1895
        a = 700 # goes down
        b = 990 # goes right
        coin = image[a:a+60, b:b+60]
        ax1 = fig.add_subplot(2, 4, 1)# image
        ax2 = fig.add_subplot(2, 4, 2)# window
        ax3 = fig.add_subplot(2, 4, 3)# self xcorrelation
        ax4 = fig.add_subplot(2, 4, 4)# fft
    if i == 1:
        scalebar =  ScaleBar(1.49253731, 'nm') # 1 461-353=67 pixels are 100 nm
        ft_scalebar = ScaleBar( 0.000327308257, 'nm') # total pixels 2047 image length = 2047*1.49253731 fftpixel = 1/1895
        a = 600 #395 # goes down 750
        b = 700 #505 # goes right 850
        coin = image[a:a+20, b:b+20]
        ax1 = fig.add_subplot(2, 4, 5)# image
        ax2 = fig.add_subplot(2, 4, 6)# window
        ax3 = fig.add_subplot(2, 4, 7)# self xcorrelation
        ax4 = fig.add_subplot(2, 4, 8)# fft
        
    result = match_template(image, coin)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]


    ax1.imshow(image, cmap=plt.cm.gray, aspect="auto")
    ax1.set_axis_off()
    # ax1.set_title('image')
    # highlight matched region
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.add_artist(scalebar)
    
    ax2.imshow(coin, cmap=plt.cm.gray, aspect="auto")
    ax2.set_axis_off()
    # ax2.set_title('template')

    xCorrImg = np.abs(result)
    xCorrImg = gaussian_filter(xCorrImg, sigma=25)
    f = ax3.imshow(xCorrImg,  aspect="auto")
    ax3.set_axis_off()
    # ax3.set_title('match_template\nresult')
    # highlight matched region
    ax3.autoscale(False)
    # ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)


    wimage = image * window('hann', image.shape) # Window is a 2d gussian shape so that the FFT will apply for mostly the center of the image
    im_f_mag = fftshift(np.abs(fftn(wimage)))

    # im_f_mag[709-2:709+2, 1024-2:1024+2] = 0 # Removing the DC becaus it is so high that it make everything look too dark
    ax4.imshow(im_f_mag, norm=LogNorm(), cmap='magma', aspect="auto")
    ft_scalebar = ScaleBar( 0.000527616141, 'nm')
    # ax4.set_title('fft')
    ax4.add_artist(ft_scalebar)
    ax4.set_axis_off()
plt.show()