# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:15:43 2024

Erik Oslen wrote the first itteration of this iSCAT image simulation script. 

Matt Kowal edited code with the following intent:
    1. Simulate iSCAT images which are representative of our experimental ratiometric images
    2. Train a NN which predicts the position of our experimental data
    

Deeptrack Documentation
https://deeptrack-20.readthedocs.io/en/latest/installation.html

Deeptrack PyPi
https://pypi.org/project/deeptrack/

Deeptrack Github
https://github.com/DeepTrackAI/DeepTrack2


@author: Matt
"""

''' CURRENT QUESTIONS / THINGS TO FIX
1. why does is make 256,256 sized images when i set image_size to 512 ???!?
2. sampletot.update()() doesnt work when using multiple particles
3. what are sample2 and sample3 used for? the comments say labels and masks, but sample3 looks more like a label to me


Things I think I should tweak:
    1. Add a mottled sort of background to match the coverslip roughness
    2. DONExxxAdd some random level of interference waves in the background (see: r8[692])
    3. Tone down the intensity of the rings so they match the milder AOBD pattern
    4. DONETHISSEEMSOKNOWxxxMake the center of the simulated particles more circular

Text header created using:
    Text to ASCII Art Generator (TAAG)
    http://www.patorjk.com/software/taag/
    Font: Slant
    Char Width: Fitted
    Char Height: Fitted
'''

import deeptrack as dt
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import skimage.measure
from skimage.filters import gaussian 


'''load experimental data '''
import pickle
with open('C:/Users/Matt/Desktop/EPD-iSCAT NTA/r8.pkl', 'rb') as f:
    r8 = pickle.load(f)
with open('C:/Users/Matt/Desktop/EPD-iSCAT NTA/constants.pkl', 'rb') as f:
    constants = pickle.load(f)
with open('C:/Users/Matt/Desktop/EPD-iSCAT NTA/r16.pkl', 'rb') as f:
    r16 = pickle.load(f)
    
    
'''#####################################################
    ____                     __   _                    
   / __/__  __ ____   _____ / /_ (_)____   ____   _____
  / /_ / / / // __ \ / ___// __// // __ \ / __ \ / ___/
 / __// /_/ // / / // /__ / /_ / // /_/ // / / /(__  ) 
/_/   \__,_//_/ /_/ \___/ \__//_/ \____//_/ /_//____/  
                                                       
#####################################################'''
#!!!
def zscore(image):
    #zero center and divide by standard deviation
    image = np.real(image)
    
    mean = np.mean(image)
    mean = 1 #shortcut/assumption
    std  = np.std(image.astype(np.float32)) #convert to a floating point for this to avoid precision overflow of 16 bit floating point
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

''' plotting functions '''
#!!!
def plot_sim_exp(sim, exp, plot_histograms=False, downsize=False):
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
    print("radius:         ", image.get_property("radius"))
    print("image stddev:   ", np.std(imagesim))
    print("section stddev: ", np.std(imagesim[0:29, 0:29]))
    print("pixel size:     ", image.get_property("pixel_size"))

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

'''#####################################################################
    _  _____  ______ ___   ______   _                              
   (_)/ ___/ / ____//   | /_  __/  (_)____ ___   ____ _ ____ _ ___ 
  / / \__ \ / /    / /| |  / /    / // __ `__ \ / __ `// __ `// _ \
 / / ___/ // /___ / ___ | / /    / // / / / / // /_/ // /_/ //  __/
/_/ /____/ \____//_/  |_|/_/    /_//_/ /_/ /_/ \__,_/ \__, / \___/ 
                           __   __                 _ /____/        
      _____ __  __ ____   / /_ / /_   ___   _____ (_)_____         
     / ___// / / // __ \ / __// __ \ / _ \ / ___// // ___/         
    (__  )/ /_/ // / / // /_ / / / //  __/(__  )/ /(__  )          
   /____/ \__, //_/ /_/ \__//_/ /_/ \___//____//_//____/           
         /____/                                                   
###################################################################'''         
#!!!

#number of particles
MULTIPARTICLE    =  False
if MULTIPARTICLE == False: num_p = 1 #number of particles
if MULTIPARTICLE == True:  num_p = np.random.randint(1,8) # pick a random number of particles


#Parameters for the simulation
image_size             = 256#512
offset_px              = 16
#radius_range = np.array([10,20])*1e-9 #use this if you pretty much only want to see noise, note: you may get erros if you pick a size thats too small
radius_range           = np.array([42,150])*1e-9 #i thought 20/30-100/160 looked good before but now idk
refractive_index_range = np.array([1.43, 1.59])
z_range_px             = np.array([-1, -2]) #-1, 4 was ok

#Setting up the optical system
#horisontal_coma_range = np.array([-100, 100]) 
NA               = 1.3
working_distance = 0.17e-3
wavelength       = 532e-9
resolution       = 340e-9
magnification    = 8.5

#other paramaters
noisemin         = 0.005
noisemax         = 0.007
clipmin, clipmax = constants["clipmin"], constants["clipmax"] 
print("clipmin, clipmax ", clipmin, clipmax)


#define a particle
particle=dt.MieSphere(
    position                = lambda: (np.random.uniform(offset_px,image_size - offset_px), np.random.uniform(offset_px,image_size - offset_px)),
    radius                  = lambda: np.random.uniform(radius_range[0], radius_range[1]),
    refractive_index        = lambda: np.random.uniform(refractive_index_range[0], refractive_index_range[1]), 
    z                       = lambda: np.random.uniform(z_range_px[0], z_range_px[1])*2,
    position_objective      = (np.random.uniform(-250,250)*1e-6,np.random.uniform(-250,250)*1e-6, np.random.uniform(-15,15)*1e-6),
    position_unit           = "pixel",
    refractive_index_medium = 1.33,
    #intensity = lambda: np.random.randint(1,100)
    )
particle_z=dt.MieSphere(
    position                = lambda: (np.random.uniform(offset_px,image_size - offset_px), np.random.uniform(offset_px,image_size - offset_px)),
    radius                  = lambda: np.random.uniform(radius_range[0], radius_range[1]),
    refractive_index        = lambda: np.random.uniform(refractive_index_range[0], refractive_index_range[1]), 
    z                       = 1,
    position_objective      = (0,0,0),
    position_unit           = "pixel",
    refractive_index_medium = 1.33,
    #intensity = lambda: np.random.randint(1,100)
    )

#define an optical system
optics=dt.Brightfield(
    NA                  = NA,
    working_distance    = working_distance,
    aberration          = dt.Astigmatism(coefficient=5),
    wavelength          = wavelength,
    resolution          = resolution,
    magnification       = magnification,
    output_region       = (0,0,image_size,image_size),
    padding             = (image_size//2,) * 4,
    polarization_angle  = lambda: np.random.rand() * 2 * np.pi,
    return_field        = True,
    backscatter         = True,
    illumination_angle  = np.pi,
    )

#filters
mask          = dt.Lambda(to_mask, mask_size=3)                # ??
make_gaussian = dt.Lambda(function=gaussian_filter, width=9)   # gaussian noise
to_real_dt    = dt.Lambda(to_real)                             # ??
dst           = dt.Lambda(ds, factor=2)                       # downsample image with skimage.measure.block_reduce(image, blocksize, func, cval)
wave          = dt.Lambda(generate_sine_wave_2D, p=lambda: np.random.uniform(0.0001,0.0009)) # This creates a wavey background emulating stray interference patterns from the optics
phadd         = dt.Lambda(phase_adder, ph=lambda: np.random.uniform(0,2*np.pi)) # ??
conj          = dt.Lambda(conjugater)                          # ??
gauss_noise   = dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax))

'''s0 is a microscop object: <deeptrack.optics.Microscope object
   sample is a deeptack feature chain: <deeptrack.features.Chain object'''
#Main particle
if MULTIPARTICLE==True:
    s0 = optics(particle^num_p)
    s1 = optics(particle_z^num_p)
if MULTIPARTICLE==False: s0 = optics(particle)
print(num_p)

#s0 = optics(particle)
#eriks version
#sample = s0 >> conj >> phadd >> wave >> conj >> dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#minimal version
#sample = s0 >> dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#epd-iscat replica 1
#sample = s0 >> wave >> dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#epd-iscat replica 2 - not sure its noticably different than replica 1

#image of multiple particles
sample = s0 >> phadd >> wave >> gauss_noise >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
image  = sample.resolve()
plt.imshow(image)
plt.show()

#sample_no_ds = s0 >> phadd >> wave >> gauss_noise


# def circle(radius=3):
#     def inner(image):
#         pos = image.property("position", get_one=False)
#         circle_mask = image
#         return circle_mask
#     return inner
# circle_mask = dt.Lambda(circle)





# #label image of multiple particles at z=1
sample2 = dt.Bind(s0, z=1, position_objective=(0,0,0)) >> phadd >> dst
image2  = sample2.resolve()
plt.imshow(image2)
plt.title("sample2")
plt.show()

#mask? = gaussian mask
sample3 = dt.Bind(s0, z=1, position_objective=(0,0,0)) >> mask >> make_gaussian >> dst
#sample3 = s1 >> mask >> make_gaussian >> dst
#sample3 = s0 >> mask >> dst
image3  = sample3.resolve()
plt.imshow(image3)
plt.show()

from skimage import draw
def circle():
    def inner(image):
        #print("circles ", image.shape)
        new_image=np.zeros(image.shape)
        positions = image.get_property("position", get_one=False)
        #print(positions)
        for p in positions:
            #print(p)
            x, y = int(p[0]), int(p[1])
            #print(x, y)
            rr, cc = draw.disk([x, y], radius=5, shape=image.shape)
            new_image[rr, cc] = 1
        return new_image
    return inner
circle_mask = dt.Lambda(circle)

# mask 2 - circle mask
sample4 = s0 >> phadd >> circle_mask >> dst
image4  = sample4.resolve()
plt.imshow(image4)
plt.title("sample4")
plt.show()

sampletot=sample&sample2&sample3&sample4
plot_labels(sampletot, titles=["sim", "label", "mask1", "mask2"])
#!!!
#%%

''' plot a simulated image next to an experimental image '''
''' here you can see what multiple simulated particles look
like even though we only feed single particles into the model '''
MULTIPARTICLE    =  True
if MULTIPARTICLE == False: num_p = 1 #number of particles
if MULTIPARTICLE == True:  num_p = np.random.randint(1,8) # pick a random number of particles

s1 = optics(particle_z^num_p)
sample_n = s1 >> phadd >> wave >> gauss_noise >> dst

plot_sim_exp(sample_n, r16, plot_histograms=False, downsize=True)


#%%


''' INITIALIZE A MODEL '''
import tensorflow
import tensorflow.keras.layers as layers
def make_Unet_model():
    activation = lambda x: layers.LeakyReLU(0.15)(x)
    model = dt.models.UNet(
        input_shape = (None, None, 1),                            # shape of the input
        conv_layers_dimensions = (16, 32, 64, 128), # number of features in each convolutional layer
        deconv_layers_dimensions = (16, 32, 64, 128),
        base_conv_layers_dimensions = (128, 128),                  # number of features at the base of the unet
        output_conv_layers_dimensions = (64, 64),               # number of features in convolutional layer after the U-net
        steps_per_pooling = 2, #2                               # number of convolutional layers per pooling layer
        number_of_outputs = 3,                                  # number of output features
        activation=activation
    )
    
    opt = tensorflow.keras.optimizers.Adam(
        learning_rate=8e-4)#, beta_1=0.9, epsilon=1e-07)
    model.compile(optimizer=opt, loss = 'mae', metrics=['mse', 'mae', 'mape', 'accuracy'])
    model.summary()
    return model
model = make_Unet_model()
#%%
import os
basepath = 'MODELS/unet'
#model_name = "iscat 1particle label mask v1"
#model_name = "iscat single-particle v3 - new mask"
model_name = "iscat dt v1"
weights_path = os.path.join(basepath, "weights", (model_name+".hdf5"))
weights_path2 = os.path.join(basepath, "weights", (model_name+".h5"))
model_path = os.path.join(basepath, "saved-models", model_name)
print(model_path)
#%% 

''' LOAD PREVIOUS WEIGHTS '''
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded previous weights: OK!")


#%%
def make_label(image, include_gauss = True):
    #concatenate the images
    #return np.concatenate((np.abs(image[1]-1)/np.std(image[0]), np.abs(image[2])), axis=2)
    #return np.abs(image[1]-1)/np.std(image[0])
    return np.concatenate( (np.abs(zscore(image[1])), np.abs(image[2]), np.abs(image[3])), axis=2)
    
''' TRAIN THE MODEL '''
reps = 10
#reps = 3

for k in range(reps):
    print("\t NOW WORKING ON Epoch: ", k)
    s = 256
    #s = 5
    data = []
    for i in range(s):
        data.append(sampletot.update()())
    data = np.array(data)
    #print(type(data[0]), data[0].shape, data[0].dtype)
    
    # GENERATE INPUT (X)
    inputs = []
    for i in range(s):
        inputs.append(zscore(data[i][0]))
    inputs = np.array(inputs)
    #print(type(inputs[0]), inputs[0].shape, inputs[0].dtype)

    # GENERATE LABELS (Y)
    labels = []
    for i in range(s):
        labels.append(make_label(data[i]))
    labels = np.array(labels)
    #print(type(labels[0]), labels[0].shape, labels[0].dtype)



    history = model.fit(inputs, labels, epochs=20, batch_size=16, shuffle=True)

    gc.collect()
    
#%%
''' generate training statistics '''
'''??????????????????? trying to make roc curve '''
plt.plot(history.history["mae"])
plt.title("Mean Absolute Error")
plt.show()
plt.plot(history.history["mse"])
plt.title("Mean Squared Error")
plt.show()
plt.plot(history.history["mape"])
plt.title("Mean Absolute Percentage Error")
plt.show()

plt.plot(history.history['accuracy'])
plt.title("Accuracy")
plt.show()



# #generate roc data
# s = 5
# test_data = []
# for i in range(s):
#     test_data.append(sampletot.update()())
# test_data = np.array(test_data)
# #print(type(data[0]), data[0].shape, data[0].dtype)

# # GENERATE INPUT (X)
# test_inputs = []
# for i in range(s):
#     test_inputs.append(zscore(test_data[i][0]))
# test_inputs = np.array(test_inputs)
# #print(type(inputs[0]), inputs[0].shape, inputs[0].dtype)

# # GENERATE LABELS (Y)
# test_labels = []
# for i in range(s):
#     test_labels.append(make_label(test_data[i]))
# test_labels = np.array(test_labels)

# from sklearn.metrics import roc_curve
# test_preds = model.predict(test_inputs).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, test_preds)



#%%
''' SAVE MODEL AND WEIGHTS '''
model.save(os.path.join(basepath, "saved-models", model_name)) #this is the one thats loaded by the particle finder
model.save(os.path.join(basepath, "weights", (model_name+".h5"))) 
model.save_weights(os.path.join(basepath, "weights", (model_name+".hdf5")) )
 
#%%
''' TEST THE MODEL ON EXPERIMENTAL DATA '''
def import_random_exp_image(r16): #r16 is a 16 bit ratiometric video
    fnum = np.random.randint(0,len(r16))       # pick a random frame
    imageexp = r16[fnum]    #use that frame as the experimental data
    # imageexp[0,0] = 1  #this is all weird non-image data from the camera
    # imageexp[0,1] = 1  #this is all weird non-image data from the camera
    # imageexp[0,2] = 1  #this is all weird non-image data from the camera
    # imageexp[0,3] = 1  #this is all weird non-image data from the camera
    return imageexp

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


'''reformat a raw 16-bit ratiometric iscat image to fit the model input'''
ratio_image = import_random_exp_image(r16)
processed_image = process_exp_image(ratio_image)
print("processed image info:")
print(processed_image.shape)
print(type(processed_image))
print(processed_image.dtype)

''' test experimental data prediciton '''
preds = model.predict(processed_image)

''' plot the experimental data prediciton '''
ppp = preds[0] #since we only used one input, our prediction array is only 1 unit long so take the first one
#test image
fig,ax=plt.subplots(2, 2,figsize=(9,9), dpi=300)
q1 = ax[0,0].imshow(ratio_image)
ax[0,0].set_axis_off()
ax[0,0].set_title("ratio")
plt.colorbar(q1, shrink=0.8)
#label
im=ppp[:,:,0]
q2 = ax[0,1].imshow(im)
ax[0,1].set_axis_off()
ax[0,1].set_title("label")
plt.colorbar(q2, shrink=0.8)
#mask1
im=ppp[:,:,1]
q2 = ax[1,0].imshow(im)
ax[1,0].set_axis_off()
ax[1,0].set_title("label")
plt.colorbar(q2, shrink=0.8)
#mask2
im=ppp[:,:,2]
q2 = ax[1,1].imshow(im)
ax[1,1].set_axis_off()
ax[1,1].set_title("mask2")
plt.colorbar(q2, shrink=0.8)
plt.tight_layout()
plt.show()


''' run a blob finder on the mask image '''
from skimage.feature import blob_dog, blob_log, blob_doh
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

positions = predicted_positions(ratio_image, ppp, output_layer=2, blob_method=2)
print("here are the positions\n", positions)



