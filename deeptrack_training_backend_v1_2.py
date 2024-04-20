# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:49:48 2024

@author: user1
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import gaussian 
# import skimage.measure
from skimage.feature import blob_dog, blob_log, blob_doh

'''#####################################################
            DEEPTRACK SIMULATION FUNCTIONS
#####################################################'''
#!!!

def zscore(image):
    #zero center and divide by standard deviation
    image = np.real(image)    
    mean  = np.mean(image)
    mean  = 1 #shortcut/assumption
    std   = np.std(image.astype(np.float32)) #convert to a floating point for this to avoid precision overflow of 16 bit floating point
    zscore_image = (image-mean)/std
    # print(mean)
    # print(std)
    # print(np.mean(zscore_image))
    # print(np.std(zscore_image))
    return zscore_image
    
def get_positions(image): 
    return np.array(image.get_property("position", get_one=False))

def get_z_positions(image):
    return np.array(image.get_property("z", get_one=False))

def get_size(image):
    return np.array(image.get_property("radius"))#, get_one=False)

# normalization function :: ERIKS CODE, CAN DELETE
# def batch_function(image):
#     return zscore(image)
#     #return np.real((image[0]-1)/np.std(image[0])) # removed abs

def generate_label(image, include_gauss = True):
    #concatenate the images
    #return np.concatenate((np.abs(image[1]-1)/np.std(image[0]), np.abs(image[2])), axis=2)
    if include_gauss:
        return np.concatenate((np.abs(image[1]-1)/np.std(image[0]), np.abs(image[2])), axis=2)
    else:
        return np.abs(image[1]-1)/np.std(image[0])

def gaussian_filter(width):
    def apply_guassian_filter(image):
        NORM_FACTOR = 100
        image = np.pad(image, [(1, 1), (1, 1), (0, 0)])
        image = gaussian(image, width, truncate=2.5) * NORM_FACTOR
        image = image[1:-1, 1:-1]
        return image
    return apply_guassian_filter

def to_real():
    def inner(image):
        return np.real(image)
    return inner

#this is a downsampling function but im going to get rid of it
def ds(factor=2):
    def inner(image):
        return skimage.measure.block_reduce(image, (factor, factor, 1), np.mean)
    return inner

def to_mask(mask_size=2):
    def inner(image):
        new_image=np.zeros(image.shape)
        pos = image.get_property("position")
        x = int(pos[0])
        y = int(pos[1])
        new_image[x-mask_size//2:x+mask_size//2,y-mask_size//2:y+mask_size//2]=1
        return new_image
    return inner

def generate_sine_wave_2D(p):
    """
    Generate a 2D sine wave pattern with adjustable direction.

    Parameters:
    - N: The size of the square image (N x N).
    - frequency: The frequency of the sine wave.
    - direction_degrees: The direction of the wave in degrees.

    Returns:
    - A 2D numpy array representing the sine wave pattern.
    """
    def inner(image):
        N = image.shape[0]
        frequency = np.random.uniform(4, 10)
        direction_degrees = np.random.uniform(0,180) # changed
        warp_factor = np.random.uniform(0, 0.5) # changed
        
        x = np.linspace(-np.pi, np.pi, N)
        y = np.linspace(-np.pi, np.pi, N)

        # Convert direction to radians
        direction_radians = np.radians(direction_degrees)

        # Calculate displacement for both x and y with warping
        warped_x = x * np.cos(direction_radians) + warp_factor * np.sin(direction_radians * x)
        warped_y = y * np.sin(direction_radians) + warp_factor * np.sin(direction_radians * y)

        # Generate 2D sine wave using the warped coordinates
        sine2D = np.sin((warped_x[:, np.newaxis] + warped_y) * frequency)# 128.0 + (127.0 * np.sin((warped_x[:, np.newaxis] + warped_y) * frequency))
        # sine2D = sine2D / 255.0

        #flip or mirror the pattern
        if np.random.rand()>0.5:
            sine2D=np.flip(sine2D,0)
        if np.random.rand()>0.5:
            sine2D=np.flip(sine2D,1)
        if np.random.rand()>0.5:
            sine2D=np.transpose(sine2D)
        image = image + np.expand_dims(sine2D, axis = -1)*p
        return image
    return inner

#PHASE ADDER??
def phase_adder(ph):
    def inner(image):
        image=image-1              #mean center at zero
        image=image*np.exp(1j*ph)  #multiples by complex exponential of a random phase angle... why tho? float->complex
        image=image+1              #re-centers data at 1
        return np.abs(image)       #return absolute value of image, and converts the complex128 image to float64
    return inner

#CONJUGATOR??
def conjugater():
    def inner(image):
        if np.random.rand()>0.5:     #there is a 50% chance this will occur
            image=np.conj(image-1)   #take the complex conjugate of a zero-centered image
            image=image+1            # re-center it at 1
            return image
        else:
            return image
    return inner







''' ############################

    DATA PREPROCESSING FUNCTIONS 

    ##########################'''

def process_exp_image(image, downsize=False): #r16 is a 16 bit ratiometric video
    image[0,0] = 1  #this is all weird non-image data from the camera
    image[0,1] = 1  #this is all weird non-image data from the camera
    image[0,2] = 1  #this is all weird non-image data from the camera
    image[0,3] = 1  #this is all weird non-image data from the camera    
    image_out = zscore(image)
    x, y = image_out.shape
    print(x, y)
    image_out = np.reshape(image_out, [x, y, 1])
    image_out = np.array([image_out])
    return image_out





''' #############################

        PLOTTING FUNCTIONS 
        
    #############################'''
#!!!
def plot_sim_exp(sim, exp, clipmin, clipmax, image_size, plot_histograms=False, downsize=False):
    r16 = exp
    sample = sim
    
    # compared simulated data to experimental data
    print("\nExperimental Data Specs:\n")
    print("--------------------")
    #fnum = -1*np.random.randint(0,100)      # use this if you watn to pick a random frame of just noise
    #fnum=45#343, 45 #this is a weird one where the stddev overflows
    fnum = np.random.randint(0,len(r16))       # pick a random frame
    imageexp = r16[fnum]    #use that frame as the experimental data
    print("imageexp", imageexp.shape)
    imageexp[0,0] = 1  #this is all weird non-image data from the camera
    imageexp[0,1] = 1  #this is all weird non-image data from the camera
    imageexp[0,2] = 1  #this is all weird non-image data from the camera
    imageexp[0,3] = 1  #this is all weird non-image data from the camera
    imageexp_clip = np.clip(imageexp, clipmin, clipmax)  #make a clipped version
    imageexp_zscore = zscore(imageexp)                   #make a zscore version
    print("Experimental Images length: ", len(r16))
    print("Frame number used: ", fnum)
    print("Frame max value: ", np.max(imageexp))
    print("Frame min value: ", np.min(imageexp))
    #print("sorted image value:\n\t", np.sort(imageexp.flatten()))
    #print(imageexp[0:10, 0:10])
    
    
    #simulated data
    print("\nSimulated Data Specs:\n")
    print("--------------------")
    im=sample.update()()
    imagesim = np.real(im)
    print("imagesim", imagesim.shape)
    if downsize==False: imagesim = np.reshape(imagesim, [image_size, image_size])
    if downsize==True:  imagesim = np.reshape(imagesim, [image_size//2, image_size//2])
    print("imagesim", imagesim.shape)
    imagesim_clip = np.clip(imagesim, clipmin, clipmax)
    imagesim_clip[0,0] = clipmin
    imagesim_clip[0,1] = clipmax
    imagesim_zscore = zscore(imagesim)
    print("position:       ", im.get_property("position"))
    print("z-position:     ", im.get_property("z"))
    print("radius:         ", im.get_property("radius"))
    print("image stddev:   ", np.std(imagesim))
    print("section stddev: ", np.std(imagesim[0:29, 0:29]))
    print("pixel size:     ", im.get_property("pixel_size"))

    #plotting function
    gx = 2 #grid size
    gy = 3
    fig,ax=plt.subplots(gy,gx,figsize=(10,10), dpi=300)
    plt.rc('font', size=10)
    plt.rc('figure', titlesize=10)
    plt.rc('ytick', labelsize=10)
    q1 = ax[0,0].imshow(imagesim)
    q2 = ax[0,1].imshow(imageexp)
    q3 = ax[1,0].imshow(imagesim_clip)
    q4 = ax[1,1].imshow(imageexp_clip)
    q5 = ax[2,0].imshow(imagesim_zscore)
    q6 = ax[2,1].imshow(imageexp_zscore)
    plt.colorbar(q1, shrink=0.82)
    plt.colorbar(q2, shrink=0.82)
    plt.colorbar(q3, shrink=0.82)
    plt.colorbar(q4, shrink=0.82)
    plt.colorbar(q5, shrink=0.82)
    plt.colorbar(q6, shrink=0.82)
    #Make Titles, Embelishments, and Finishing Touches
    ax[0,0].set_xlabel("Deeptrack Sim")
    ax[0,1].set_xlabel(("iSCAT frame "+str(fnum)))
    ax[1,0].set_xlabel("Deeptrack - Scaled")
    ax[1,1].set_xlabel(("iSCAT - Scaled "+str(fnum)))
    ax[2,0].set_xlabel("Deeptrack - zscore")
    ax[2,1].set_xlabel(("iSCAT - zscore "+str(fnum)))
    ax[0,0].set_ylabel("Raw Image", size=20)
    ax[1,0].set_ylabel("clipped image", size=20)
    ax[2,0].set_ylabel("Z-score", size=20)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[2,0].set_xticks([])
    ax[2,0].set_yticks([])
    ax[2,1].set_xticks([])
    ax[2,1].set_yticks([])
    fig.suptitle("simulated image                                          experimental images")
    plt.tight_layout()
    plt.show()
    
    
    #do you want to plot histograms for the data?
    if plot_histograms:
        #plot histograms
        fig,ax=plt.subplots(gy,gx,figsize=(6,6), dpi=300)
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 2}
        SMALL_SIZE = 12
        MEDIUM_SIZE=18
        BIGGER_SIZE=8
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE) 
        ax[0,0].hist(imageexp)
        ax[0,1].hist(imagesim)
        ax[1,0].hist(imageexp_clip)
        ax[1,1].hist(imagesim_clip)
        ax[2,0].hist(imageexp_zscore)
        ax[2,1].hist(imagesim_zscore)
        ax[0,0].set_yticks([])
        ax[0,1].set_yticks([])
        ax[1,0].set_yticks([])
        ax[1,1].set_yticks([])
        ax[2,0].set_yticks([])
        ax[2,1].set_yticks([])
        fig.suptitle("simulated histograms                                          experimental histograms")
        #plt.tight_layout()
        plt.show()
    return

# check each input next to each other for one sample
def plot_labels(sampletot, titles):
    sample_bundle=sampletot
    input_data = sample_bundle.update()()
    # in1 = np.real(input_data[0])
    # in2 = np.real(input_data[1])
    # in3 = np.real(input_data[2])
    fig, axs = plt.subplots(1, len(titles), figsize=(15, 5))
    plt.rc('font', size=18)
    for i in range(len(titles)):
        #print(i, input_data_idx)
        in_data = np.real(input_data[i])
        axs[i].imshow(in_data)
        axs[i].axis('off')  # Optional: Turn off axis
        axs[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()







def make_label(image, include_gauss = True):
    #concatenate the images
    #return np.concatenate((np.abs(image[1]-1)/np.std(image[0]), np.abs(image[2])), axis=2)
    #return np.abs(image[1]-1)/np.std(image[0])
    return np.concatenate( (np.abs(zscore(image[1])), np.abs(image[2]), np.abs(image[3])), axis=2)
    




def import_random_exp_image(r16): #r16 is a 16 bit ratiometric video
    fnum = np.random.randint(0,len(r16))       # pick a random frame
    imageexp = r16[fnum]    #use that frame as the experimental data
    # imageexp[0,0] = 1  #this is all weird non-image data from the camera
    # imageexp[0,1] = 1  #this is all weird non-image data from the camera
    # imageexp[0,2] = 1  #this is all weird non-image data from the camera
    # imageexp[0,3] = 1  #this is all weird non-image data from the camera
    return imageexp




''' run a blob finder on the mask image '''
def predicted_positions(ratio_image, pred_images, output_layer=2, blob_method=0):
    pred_image = ppp[:,:,output_layer]
    
    #find blobs
    if blob_method==0:
        blobs_log = blob_log(pred_image, max_sigma=10, num_sigma=10, threshold=.1)
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
        log = blobs_log
        positions = [[x[0], x[1]] for x in blobs_log]
    if blob_method==1:
        blobs_dog = blob_dog(pred_image, max_sigma=10, threshold=.05)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
        log = blobs_dog
        positions = [[x[0], x[1]] for x in blobs_dog]
    if blob_method==2:
        blobs_doh = blob_doh(pred_image, max_sigma=10, threshold=.001)
        log = blobs_doh
        positions = [[x[0], x[1]] for x in blobs_doh]
    
    #make plot
    #print("test\n",positions)
    fig, ax = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    
    #experimental image
    ax[0,0].imshow(ratio_image)
    ax[0,0].set_axis_off()
    ax[0,0].set_title("exp")
    
    # marked blobs
    ax[0,1].imshow(pred_image)
    ax[0,1].set_axis_off()
    for idx, blob in enumerate(positions):
        y, x, r = log[idx]
        c = plt.Circle((x, y), 10, color="red", linewidth=2, fill=False, alpha=0.5)
        ax[0,1].add_patch(c)
    ax[0,1].set_title("labeled blobs")
    
    #predicted image
    ax[1,0].imshow(pred_image)
    ax[1,0].set_axis_off()
    ax[1,0].set_title("predicted label")
    
    #found blobs
    new_image=np.zeros(([256,256]))
    for p in positions:
        x, y = int(p[0]), int(p[1])
        #print(p, x, y)
        rr, cc = draw.disk([x, y], radius=6, shape=[256,256])
        new_image[rr, cc] = 255
    ax[1,1].imshow(new_image)
    ax[1,1].set_axis_off()
    ax[1,1].set_title("found blobs")
    plt.tight_layout()
    plt.show()
    
    return positions