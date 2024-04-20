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
    
installation:
    pip install matplotlib
    pip install numpy
    pip install scikit-image
    pip install deeptrack
'''

import deeptrack as dt
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gc
# 

# 




''' initialize current working directory ''' #and load iscat-yolo backend '''
script_dir, script_filename = os.path.split(__file__)  # get folder and filename of this script
expdata_dir = os.path.join(os.path.split(script_dir)[0], 'iscat-dt-data')
sys.path.insert(0, script_dir)               # add the current folder to the system path
import deeptrack_training_backend_v1_2 as dttrain
#iscatdt.loadscreen()




'''load experimental data '''

with open(os.path.join(expdata_dir, 'r8.pkl'), 'rb') as f:
    r8 = pickle.load(f)
with open(os.path.join(expdata_dir, 'constants.pkl'), 'rb') as f:
    constants = pickle.load(f)
with open(os.path.join(expdata_dir, 'r16.pkl'), 'rb') as f:
    r16 = pickle.load(f)
    





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
# MULTIPARTICLE    =  False
# if MULTIPARTICLE == False: num_p = 1 #number of particles
# if MULTIPARTICLE == True:  num_p = np.random.randint(1,8) # pick a random number of particles


''' DeepTrack Simulation Parameters '''
#Image parameters
image_size             = 256#512
offset_px              = 16
#radius_range = np.array([10,20])*1e-9 #use this if you pretty much only want to see noise, note: you may get erros if you pick a size thats too small
radius_range           = np.array([42,150])*1e-9 #i thought 20/30-100/160 looked good before but now idk
refractive_index_range = np.array([1.43, 1.59])
z_range_px             = np.array([-1, -2]) #-1, 4 was ok

#Optical system parameters
#horisontal_coma_range = np.array([-100, 100]) 
NA               = 1.3
working_distance = 0.17e-3
wavelength       = 532e-9
resolution       = 340e-9
magnification    = 8.5

#Filter parameters
noisemin         = 0.005
noisemax         = 0.007
clipmin, clipmax = constants["clipmin"], constants["clipmax"] 
print("clipmin, clipmax ", clipmin, clipmax)


''' DeepTrack Object Initialization (Particle and Objective)  '''
#particle, 3d position
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


#iSCAT Optical System
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

''' Define Image Filters'''
mask          = dt.Lambda(dttrain.to_mask, mask_size=3)                # ??
make_gaussian = dt.Lambda(function=dttrain.gaussian_filter, width=9)   # gaussian noise
to_real_dt    = dt.Lambda(dttrain.to_real)                             # ??
dst           = dt.Lambda(dttrain.ds, factor=2)                       # downsample image with skimage.measure.block_reduce(image, blocksize, func, cval)
wave          = dt.Lambda(dttrain.generate_sine_wave_2D, p=lambda: np.random.uniform(0.0001,0.0009)) # This creates a wavey background emulating stray interference patterns from the optics
phadd         = dt.Lambda(dttrain.phase_adder, ph=lambda: np.random.uniform(0,2*np.pi)) # ??
conj          = dt.Lambda(dttrain.conjugater)                          # ??
gauss_noise   = dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax))



''' Generate a Simulated iSCAT Image '''
'''s0 is a microscope object: <deeptrack.optics.Microscope object '''
s0 = optics(particle)


#Multi particle
# if MULTIPARTICLE==True:
#     s0 = optics(particle^num_p)
#     s1 = optics(particle_z^num_p)
# if MULTIPARTICLE==False: s0 = optics(particle)
# print(num_p)



#eriks version
#s0 = optics(particle)
#sample = s0 >> conj >> phadd >> wave >> conj >> dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#minimal version
#sample = s0 >> dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#epd-iscat replica 1
#sample = s0 >> wave >> dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#epd-iscat replica 2 - not sure its noticably different than replica 1




''' Define an simulated iSCAT image, with noise, distortions, etc. '''
''''   iscat_sim is a deeptack feature chain: <deeptrack.features.Chain object'''
''''   iscat_image is a deeptack Image: <deeptrack.Image.Image> object'''
iscat_sim = s0 >> phadd >> wave >> gauss_noise >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
iscat_image  = iscat_sim.resolve()
plt.imshow(iscat_image)
plt.show()




''' define a label where the particle is at z=1 '''
label_sim = dt.Bind(s0, z=1, position_objective=(0,0,0)) >> phadd >> dst
label_image  = label_sim.resolve()
plt.imshow(label_image)
plt.title("Label")
plt.show()




''' Define a new mask where each particle is labelled with a 2d gaussian '''
#mask? = gaussian mask
mask1_sim = dt.Bind(s0, z=1, position_objective=(0,0,0)) >> mask >> make_gaussian >> dst
#sample3 = s1 >> mask >> make_gaussian >> dst
#sample3 = s0 >> mask >> dst
mask1_image  = mask1_sim.resolve()
plt.imshow(mask1_image)
plt.show()




''' Define a new mask where each particle is labelled with a circle '''
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

mask2_sim    = s0 >> phadd >> circle_mask >> dst
mask2_image  = mask2_sim.resolve()
plt.imshow(mask2_image)
plt.title("mask2")
plt.show()




''' Define the data that will be fed into the NN. '''
''' When it's defined this way, its possible to generate
    all four images using the same particle position.'''
sampletot = iscat_sim & label_sim & mask1_sim & mask2_sim
dttrain.plot_labels(sampletot, titles=["sim", "label", "mask1", "mask2"])
#!!!
#%%

''' PLOT SIMULATED IMAGE NEXT TO EXPERIMETNAL IAMGE '''
''' here you can see what multiple simulated particles look
like even though we only feed single particles into the model '''

MULTIPARTICLE    =  True
if MULTIPARTICLE == False: num_p = 1 #number of particles
if MULTIPARTICLE == True:  num_p = np.random.randint(1,8) # pick a random number of particles

#Particle, z = 1
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
s1 = optics(particle_z^num_p)
sample_n = s1 >> phadd >> wave >> gauss_noise >> dst

dttrain.plot_sim_exp(sample_n, r16, clipmin, clipmax, image_size, plot_histograms=False, downsize=True)


#%%


''' INITIALIZE A Unet MODEL '''
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
''' DEFINE MODEL NAME AND SAVE/LOAD PATH '''
import os
model_name = "iscat dt - garbage"
modelspath = os.path.join(script_dir, "models")
#model_name = "iscat 1particle label mask v1"
#model_name = "iscat single-particle v3 - new mask"
# weights_path  = os.path.join(modelspath, "weights", (model_name+".hdf5"))
# weights_path2 = os.path.join(modelspath, "weights", (model_name+".h5"))
# model_path    = os.path.join(modelspath, "saved-models", model_name)
#print(model_path)
 

# ''' LOAD PREVIOUS WEIGHTS '''
# if os.path.exists(weights_path):
#     model.load_weights(weights_path)
#     print("Loaded previous weights: OK!")


#%%

''' TRAIN THE MODEL '''
#slow
reps = 10
s = 256

#quick
reps = 3
s = 3

for k in range(reps):
    print("\t NOW WORKING ON Epoch: ", k)
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
model_name = "iscat dt - garbage"
model_file = os.path.join(modelspath, model_name)
model.save(os.path.join(modelspath, (model_name+".h5"))) #i think this is the one that works

#model.save(model_file) #i think this is the one thats loaded by the particle finder
#model.save(os.path.join(modelspath, "weights", (model_name+".keras"))) #seems to work ok
#model.save_weights(os.path.join(modelspath, "weights", (model_name+".hdf5")) )
#model.save_weights(os.path.join(modelspath, "weights", (model_name+".weights.h5")) )

 
#%%
''' TEST THE MODEL ON EXPERIMENTAL DATA '''



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




positions = predicted_positions(ratio_image, ppp, output_layer=2, blob_method=2)
print("here are the positions\n", positions)



