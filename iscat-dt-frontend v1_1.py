# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:46:14 2023


!!!!!!!         RUN THIS SCRIPT IN THE "Erik" ENVIRONMENT        !!!!!!!!


This script is the "front end" for the iscat-yolo methods. 

The purpose is the characterize an iscat movie and produce quantitative results
about the particles in the system.

A Chronological Order of the Methods
    Import .bin file
    
    Perform Ratiometric Particle Finding
        For Each Frame
            Generate Ratiometric Image
            Generate a "frame particle list" by finding Particles in the Ratiometric image using YOLO
            

The results produced:
    info
    info
    info
    info
    
Required Installations:
    
    pip install numpy
    pip install tqdm
    pip install scikit-image
    pip install opencv-python
    pip install matplotlib
    pip install tensorflow
    pip install pandas

@author: Matt
"""


#add the current folder to the system directory to look for modules
import sys
import os
import time
import numpy as np


# folder, filename = os.path.split(__file__)  # get folder and filename of this script
# #modfolder = os.path.join(folder)            # I cant remember why this is here
# sys.path.insert(0, folder)               # add the current folder to the system path




''' initialize current working directory and load iscat-yolo backend '''
script_dir, script_filename = os.path.split(__file__)  # get folder and filename of this script
expdata_dir = os.path.join(os.path.split(script_dir)[0], 'iscat-dt-data')
sys.path.insert(0, script_dir)               # add the current folder to the system path
import iscat_dt_backend_v1_1 as iscatdt
iscatdt.loadscreen()

# ''' step 1.5 test some thresholds'''

# from tensorflow.keras.models import load_model
# from tensorflow import keras
# constants = {}
# constants["deeptrack model loc"] = os.path.join(script_dir, "models/iscat dt - garbage.h5")
# print(constants["deeptrack model loc"])
# deeptrack_model     = load_model(constants["deeptrack model loc"])

# from tensorflow.keras.models import load_model
# dtmod = load_model("C:/Users/user1/Documents/GitHub/iscat-dt/models/iscat dt - garbage.h5")
# #deeptrack_model     = keras.layers.TFSMLayer(constants["deeptrack model loc"], call_endpoint='serving_default')



''' PLEASE SPECIFY A .BIN FILE OR AN .MP4 VIDEO FILE '''
# '''LOAD_FROM_BINFILE:'''
# bin1 = r'2023-05-26-50nm PS 0.1nM ITO pH7 Laser150mW - sq/VIDEOS/2023-05-26_16-04-32_raw_12_256_200.bin'
# bin2 = r'2022-06-30_14-56-58_raw_12_256_68.bin'
# binfile = os.path.join(expdata_dir, bin1)
# binimages = iscatdt.load_binfile_into_array(binfile)
# basepath, filename, name, nframes, fov, x, y, fps, voltfile = iscatdt.get_bin_metadata(binfile)

''' LOAD_FROM_MP4VIDEO:'''
# mp4file = os.path.join(expdata_dir, r'2023-05-26_16-04-32_raw_12_256_200.mp4')
# binimages = iscatdt.load_video(mp4file)
# basepath, filename, name, nframes, fov, x, y, fps, voltfile = iscatdt.get_bin_metadata(mp4file)


''' LOAD PREVIOUSLY PROCESSED (RATIOMETRIC ONLY) EXPERIMENTAL DATA '''
'''load experimental data '''
with open(os.path.join(expdata_dir, 'r8.pkl'), 'rb') as f:
    r8 = pickle.load(f)
with open(os.path.join(expdata_dir, 'constants.pkl'), 'rb') as f:
    constants = pickle.load(f)
with open(os.path.join(expdata_dir, 'r16.pkl'), 'rb') as f:
    r16 = pickle.load(f)

    
constants["deeptrack model loc"] = os.path.join(script_dir, "models/iscat dt - garbage.h5")
print(constants["deeptrack model loc"])
deeptrack_model     = load_model(constants["deeptrack model loc"])

#%%
test_img = r16[92].copy()
print(np.min(test_img), np.max(test_img))
plt.imshow(test_img)


min_sigma = 4
max_sigma = 10
threshold = 0.01
i16 = iscatdt.test_blob(test_img, deeptrack_model, constants, min_sigma, max_sigma, threshold)
plt.imshow(i16)
plt.title("16 bit marked")
plt.show()
#%%




#### TESTING CONDITIONS ###
#binfile = r'D:/PS - Kinetics 1/50nm PS npn sq - weird frwquencies/0.1 nM PS 50 data, probably mislabeled/2023-05-25-50nm PS 0.1nM ITO pH7 Laser150mW - sq/VIDEOS/2023-05-26_16-04-32_raw_12_256_200.bin'
#binfile = r"D:/ADDITIONAL DATA FOR ACS NANO/new PS surface saturation limit/2023-11-14-100nm PS 1.0nM ITO pH7 Laser150mW/VIDEOS/2023-11-14_15-09-42_raw_12_256_212.bin"
#binfile = r"C:/Users/user1/Documents/GitHub/iscat-dt-data/2023-05-26-50nm PS 0.1nM ITO pH7 Laser150mW - sq/VIDEOS/2023-05-26_16-04-32_raw_12_256_200.bin"

# ''''IMPORT AND PROCESS BINARY FILE'''
# running_for_first_time = True
# if running_for_first_time:
#     binimages = iscat.load_binfile_into_array(binfile)
#     basepath, filename, name, nframes, fov, x, y, fps = iscat.get_bin_metadata(binfile) # get basic info about binfile
#     # if not os.path.exists(os.path.join(basepath, "output")):
#     #     os.makedirs(os.path.join(basepath, "output"))


''' parameters '''
nm_per_px     = fov*1000/x
constants = {}
constants['sample name']        = "test" #sample_name
constants['bufsize']            = 60                   # 1/2 the ratiometric buffer size
constants['clipmin']            = 0.97                 #0.95,  # black point clipping of ratiometric image
constants['clipmax']            = 1.03                 #1.05,  # white point clipping of ratiometric image
constants['bbox']               = 15         # 1/2 the width of a saved particle image. this also defines where to cut off edge particles
constants['MIN_DIST']           = 10         # minimum distance for new overlapping particles to be considered the same      
constants['temporal tolerance'] = 10         # maximum number of skipped frames for particle to be considered the same      
constants['spatial tolerance']  = 10         # maximum distance from particle in a previous frame to be considered the same
constants['history tolerance']  = 100        # maximum number of frames to look back before giving up the search for a matching particle
constants["video x dim"]        = x
constants["video y dim"]        = y
constants["basepath"]           = basepath
constants["output path"]        = os.path.join(basepath, (name[:19] + " deeptrack " + iscatdt.getVersion()))
constants["name"]               = name
constants["fps"]                = fps
constants["fov"]                = fov
constants['output framerate']   = 50#fps #200, 24
constants["nframes"]            = nframes
constants["timestamp"]          = name[:20]
constants["frame total"]        = nframes

#deeptrack stuff
#constants["deeptrack model loc"] = os.path.join(script_dir, "model/iscat dt v1")
constants["deeptrack model loc"] = os.path.join(script_dir, "models/iscat dt - garbage.h5")

if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"])

# ########################################################
# # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
use_short_video = True
if use_short_video:
    start_frame = 9500#9000#200*500#30000#350  #2200
    nframes     = 700#1300#200*510 #12000  # 12000 frames is about a minute of video
    binimages      = binimages[start_frame:(start_frame+nframes)]
    constants["nframes"] = nframes
# else:
#     # # # OR USE THE FULL FILE
#     images = binimages
# ########################################################

# #look for weird data in the corner
# print("check out this weird data in the top right corner")
# fnum = np.random.randint(0,len(binimages))
# sample_=binimages[fnum]
# print(sample_[0:10, 0:10])


#%%
'''step 1: generate ratiometric video '''
r8, r16, noise = iscatdt.ratio_only(binimages, constants, return_v16=True)
#iscat.save_bw_video(r8, constants, "ratio", print_frame_nums = True)

#%%
import matplotlib.pyplot as plt
plt.plot(noise)
plt.title("noise")
plt.show()
#%%
''' step 1.5 test some thresholds'''
from tensorflow.keras.models import load_model
constants["deeptrack model loc"] = os.path.join(script_dir, "model/weights/iscat dt - garbage.keras")

#deeptrack_model     = load_model(constants["deeptrack model loc"])
deeptrack_model     = keras.layers.TFSMLayer(constants["deeptrack model loc"], call_endpoint='serving_default')


'''
ValueError: File format not supported: filepath=c:\users\user1\documents\github\iscat-dt\model/iscat dt v1. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(c:\users\user1\documents\github\iscat-dt\model/iscat dt v1, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).
'''
#%%
test_img = r16[92].copy()
print(np.min(test_img), np.max(test_img))
plt.imshow(test_img)


min_sigma = 4
max_sigma = 10
threshold = 0.01
i16 = iscat.test_blob(test_img, deeptrack_model, constants, min_sigma, max_sigma, threshold)


#%%
''' step 2 generate particle list from ratiometric video '''
constants["deeptrack model loc"] = "C:/Users/Matt/MODELS/unet/saved-models/iscat dt v1"
#method 1: this finds particles frame by frame
#vlabel, vmask1, vmask2, pl = iscat.deeptrack_particle_finder(r16, constants)

#method 2: (preferred) this finds particles from a whole video. may not work for huge videos
pl = iscat.deeptrack_particle_finder2(r16, constants, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)

#0
'''step 3 save particle video '''

iscat.save_particle_video(r8, pl, constants, "deeptrack-pvid")


#%%


#data
iscatdt.save_pickle_data(pl, constants, tag)
iscatdt.save_constants(constants)

#spreadsheet
iscatdt.generate_sdcontrast_csv(pl, constants, tag)
iscatdt.generate_particle_list_csv(pl, constants, tag)
iscatdt.generate_landing_rate_csv(pl, constants, tag)
iscatdt.generate_noisefloor_csv(noise, constants)
#image
iscatdt.plot_landing_map(pl, constants, tag)
iscatdt.plot_landing_rate(pl, constants, tag)
iscatdt.plot_sdcontrast_hist(pl, constants, tag)
iscatdt.plot_waterfall(pl, constants, tag)


#%%
'''old old old old old '''
'''old old old old old '''
'''old old old old old '''
'''old old old old old '''
'''old old old old old '''
'''old old old old old '''
'''old old old old old '''
'''old old old old old '''
'''old old old old old '''


constants["deeptrack model loc"] = "C:/Users/Matt/MODELS/unet/save-method/iscat single-particle v2 - new mask"

r8, r16, m16la, m16m1, m16m2, pl, results, noise = iscat.ratio_particle_finder_deeptrack(binimages, constants)

#%%
iscat.save_particle_video(r8, pl, constants, "deeptrack-pvid")

#%%

''' run a blob finder on the mask image '''
from skimage.feature import blob_dog, blob_log, blob_doh
def predicted_positions(pred_image, max_sigma=10, threshold=0.001):
    #find blobs using determinant of hessian because its the fastest
    blobdoh_pos = blob_doh(pred_image)
    positions = [[x[0], x[1]] for x in blobdoh_pos]
    return positions

positions = predicted_positions(m16la[0])
print("here are the positions\n", positions)

#%%

#%%
print(np.min(m16la), np.max(m16la))
print(np.min(m16m1), np.max(m16m1))
print(np.min(m16m2), np.max(m16m2))


import matplotlib.pyplot as plt
   
def float_to_8bit(float_image): #convert floating point to 8-bit image (0-255)
    imin = np.min(float_image)
    imax = np.max(float_image)
    image_out = (float_image - imin) / (imax - imin) * 255
    return image_out.astype(np.uint8)

def float_to_8bit_vid(float_vid):
    vid_out = np.array([float_to_8bit(f) for f in float_vid])
    return vid_out

def float_to_thresh_vid(float_vid, thresh):
    vid_out = np.where(float_vid < thresh, 0, 255)
    return vid_out.astype(np.uint8)

#convert mask images from 16 bit floats to 8 bit ints
m16la_8bit = float_to_8bit_vid(m16la)
m16m1_8bit = float_to_8bit_vid(m16m1)
m16m2_8bit = float_to_8bit_vid(m16m2)
print(np.min(m16la_8bit), np.max(m16la_8bit))
print(np.min(m16m1_8bit), np.max(m16m1_8bit))
print(np.min(m16m2_8bit), np.max(m16m2_8bit))

#convert mask images from 16 bit floats to 8 bit threshold images
m16la_thresh = float_to_thresh_vid(m16la, 3)
m16m1_thresh = float_to_thresh_vid(m16m1, 0.1)
m16m2_thresh = float_to_thresh_vid(m16m2, 0.1)
print(np.min(m16la_thresh), np.max(m16la_thresh))
print(np.min(m16m1_thresh), np.max(m16m1_thresh))
print(np.min(m16m2_thresh), np.max(m16m2_thresh))

#concatenate some videos together for comparison
#concatvid = iscat.save_multi_vid([r8, m16la_8bit, m16m1_8bit, m16m2_8bit], constants, "--multi--")
concatvid = iscat.save_multi_vid([r8, m16la_thresh], constants, "--multi--")
plt.imshow(concatvid[0])
plt.show()
iscat.save_bw_video(concatvid, constants, "--concatvid---")




#%%
video_out = iscat.save_particle_video(r8, pl, constants, tag="deeptrack")

#%%


print(type(video_out), video_out.shape, video_out.dtype)

import cv2

# VIDEO SAVING
n, x, y = r8.shape
# Generate Filename
name = constants["name"]
tag = "dt test"
#filename = name + tag + "-color.avi"
filename = name + tag + "-color.mp4"

save_file_path = os.path.join(constants["output path"], filename)

# Write and save video
print("Saving Yolo Particle Video to:   ", save_file_path)
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
                    #use a different 4-byte code for other codecs
                    #https://www.fourcc.org/codecs.php
videoObject = cv2.VideoWriter(save_file_path, fourcc, constants["output framerate"], (y, x), isColor = True)
for c in range(n):
    videoObject.write(video_out[c])
videoObject.release()


#%%
import imageio
writer = imageio.get_writer(save_file_path)

for frame in video_out:
    writer.append_data(frame)

# Close the writer
writer.close()

#%%





print(sys.getsizeof(binimages)/1000/1000, "Gbytes")
print(sys.getsizeof(r8)/1000/1000, "Gbytes")
print(sys.getsizeof(r16)/1000/1000, "Gbytes")

import pickle
with open('C:/Users/Matt/Desktop/EPD-iSCAT NTA/r8.pkl', 'wb') as f:
    pickle.dump(r8, f)
with open('C:/Users/Matt/Desktop/EPD-iSCAT NTA/constants.pkl', 'wb') as f:
    pickle.dump(constants, f)

with open('C:/Users/Matt/Desktop/EPD-iSCAT NTA/r16.pkl', 'wb') as f:
    pickle.dump(r16, f)
 

#%%
iscat.save_bw_video(r8, constants, "ratio", print_frame_nums = False)

#%%
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean

testimg = r8[0]

testimg = testimg.reshape((256,256,1))
plt.imshow(testimg)
plt.show()


inputs_mdk = np.array([downscale_local_mean(x, (4, 4)).reshape((64,64,1)) for x in r8[0:16]])
#for i in range(16):
    
#%%
'''RATIOMETRIC PARTICLE FINDER'''
r8, pl, results_list, noise = iscat.ratio_particle_finder(binimages, constants)
print("Memory Check:")
print("\n size of bin images:         ", sys.getsizeof(binimages)/1000000000, " gig")
print("\n size of ratiometric_video:  ", sys.getsizeof(r8)/1000000, " mb")
print("\n size of particle_list:      ", sys.getsizeof(pl)/1000000, " mb")
print("\n size of results_list:       ", sys.getsizeof(results_list)/1000000, " mb")

''' GENERATE OUTPUT '''
#keep only if longer than 10 frames
tag_ = "yolo-conf10-trim10" 
pl2 = []
new_id = 1
for p in pl:
    if len(p.f_vec) > 10:
        pl2.append(p)
        pl2[-1].pID = new_id
        new_id += 1
#%%
iscat.save_particle_list_video(r8, pl2, constants, tag_)
generate_output(pl2, constants, tag_)

#save_all_particle_images(pl2, constants, tag_)
#iscat.save_particle_list_video(r8, pl, constants, tag_)
#generate_output(pl2, constants, tag_)

#%%

iscat.save_si_video(r8, pl2, constants, "-si-", offset=0)

 


#%% #!!!
#        _____  _                __            __     _      
#       / ___/ (_)____   ____ _ / /___        / /_   (_)____ 
#       \__ \ / // __ \ / __ `// // _ \      / __ \ / // __ \
#      ___/ // // / / // /_/ // //  __/  _  / /_/ // // / / /
#     /____//_//_/ /_/ \__, //_/ \___/  (_)/_.___//_//_/ /_/ 
#                     /____/                                


# def i16_to_i8(i16, constants):
#     clipmin = constants["clipmin"]
#     clipmax = constants["clipmax"]
#     i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
#     return i8

# from PIL import Image
# def save_all_particle_images(pl, constants, tag):
#     print("\n\t\t PRINTING PARTICLE IMAGES...")
#     pimage_folder = os.path.join(constants["output path"], tag+"-pimage")
#     if not os.path.exists(pimage_folder): os.makedirs(pimage_folder)
    
#     yoloimage_folder = os.path.join(constants["output path"], tag+"yoloimage")
#     if not os.path.exists(yoloimage_folder): os.makedirs(yoloimage_folder)
    
    
#     new_pimage_folder = os.path.join(constants["output path"], tag+"new_pimage")
#     if not os.path.exists(new_pimage_folder): os.makedirs(new_pimage_folder)
    
    
#     for p in pl:
#         #print(type(p.yoloimage_vec[0]))
        
#         #save yolo image
#         for c, i in enumerate(p.yoloimage_vec):
#             #print(p.px_vec[c], p.py_vec[c], p.wx_vec[c], p.wy_vec[c])
#             #print(os.path.join(constants["output path"], "yoloimage", ("yoloimage-" + str(c) +".png")), type(i), i.shape)
#             im = Image.fromarray(i16_to_i8(i, constants))
#             im.save(os.path.join(yoloimage_folder, ("yoloimage-pID " + str(p.pID) +"-"+ str(c) +".png")))
        
#         #save 30x30 pimage
#         for c, i in enumerate(p.pimage_vec):
#             #print(p.px_vec[c], p.py_vec[c], p.wx_vec[c], p.wy_vec[c])
#             #print(os.path.join(constants["output path"], "pimage", ("pimage-" + str(c) +".png")), type(i), i.shape)
#             im = Image.fromarray(i16_to_i8(i, constants))
#             im.save(os.path.join(pimage_folder, ("pimage-pID " + str(p.pID) +"-"+ str(c) +".png")))
            
#         #save 30x30 new_pimage
#         for c, i in enumerate(p.new_pimage_vec):
#             #print(p.px_vec[c], p.py_vec[c], p.wx_vec[c], p.wy_vec[c])
#             #print(os.path.join(constants["output path"], "pimage", ("pimage-" + str(c) +".png")), type(i), i.shape)
#             im = Image.fromarray(i16_to_i8(i, constants))
#             im.save(os.path.join(new_pimage_folder, ("new_pimage-pID " + str(p.pID) +"-"+ str(c) +".png")))




''' PROCESS A SINGLE .BIN FILE '''
def generate_exp_data(binfile, sample_name):
    st = time.time()
    ###########################################################################
    
    
    
    
    ''''IMPORT AND PROCESS BINARY FILE'''
    running_for_first_time = True
    if running_for_first_time:
        binimages = iscat.load_binfile_into_array(binfile)
        basepath, filename, name, nframes, fov, x, y, fps = iscat.get_bin_metadata(binfile) # get basic info about binfile
        # if not os.path.exists(os.path.join(basepath, "output")):
        #     os.makedirs(os.path.join(basepath, "output"))


        
    nm_per_px     = fov*1000/x
    constants = {}
    constants['sample name']        = sample_name
    constants['bufsize']            = 20                   # 1/2 the ratiometric buffer size
    constants['clipmin']            = 0.97                 #0.95,  # black point clipping of ratiometric image
    constants['clipmax']            = 1.03                 #1.05,  # white point clipping of ratiometric image

    constants['bbox']               = 15         # 1/2 the width of a saved particle image. this also defines where to cut off edge particles
    constants['MIN_DIST']           = 10         # minimum distance for new overlapping particles to be considered the same      
    constants['temporal tolerance'] = 10         # maximum number of skipped frames for particle to be considered the same      
    constants['spatial tolerance']  = 10         # maximum distance from particle in a previous frame to be considered the same
    constants['history tolerance']  = 100        # maximum number of frames to look back before giving up the search for a matching particle

    constants["video x dim"]        = x
    constants["video y dim"]        = y
    constants["basepath"]           = basepath
    constants["output path"]        = os.path.join(basepath, (name[:19] + " yolo_out " + iscat.getVersion()))
    constants["name"]               = name
    constants["fps"]                = fps
    constants["fov"]                = fov
    constants['output framerate']   = 50#fps #200, 24
    constants["nframes"]            = nframes
    constants["timestamp"]          = name[:20]

    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train31/weights/best.pt'
    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train32/weights/best.pt'
    constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train37/weights/best.pt'
    constants["min confidence"]         = 0.45 # good for contrast accuracy     # 0.3 #good compromise #0.05 # good for exploratory dialing back


    if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"])

   
   

    # ########################################################
    # # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
    use_short_video = False
    if use_short_video:
        start_frame = 800#30000#350  #2200
        nframes     = 1000 #12000  # 12000 frames is about a minute of video
        binimages      = binimages[start_frame:(start_frame+nframes)]
        constants["nframes"] = nframes
    # else:
    #     # # # OR USE THE FULL FILE
    #     images = binimages
    # ########################################################
       

    '''RATIOMETRIC PARTICLE FINDER'''
    r8, pl, results_list, noise = iscat.ratio_particle_finder(binimages, constants)
    iscat.generate_noisefloor_csv(noise, constants)
    # intitally: at this point (early version), one minute of "dense" particle video (i.e ~20 particles/frame). It takes about 6 minutes (~30it/s). It takes 22 minutes to process a 3 minute video
    # then: somehow this changed and exploded to 90 minutes for a 3 minute video. this happened after i added all the particle bookkeeping stuff, but when i record the time each process takes it seems that the only thing that gets slower over time is the yolo particle finder. so now im not sure what to think about this
    # finally: I now run this program in the conda yolo environment so that it runs on the gpu and its not blazingly (relative) fast (60it/s). A 5 minute video takes around 20 min
    print("Memory Check:")
    print("\n size of bin images:         ", sys.getsizeof(binimages)/1000000000, " gig")
    print("\n size of ratiometric_video:  ", sys.getsizeof(r8)/1000000, " mb")
    print("\n size of particle_list:      ", sys.getsizeof(pl)/1000000, " mb")
    print("\n size of results_list:       ", sys.getsizeof(results_list)/1000000, " mb")


      
    ''' GENERATE OUTPUT '''
    
    
    
        
    #tag_ = "yolo-c45"   
    #iscat.save_particle_list_video(r8, pl, constants, tag_)
    #iscat.save_bw_video(r8, constants, "ratiometric", print_frame_nums=False)
    #generate_output(pl, constants, tag_)
    #save_all_particle_images(pl, constants, tag_)
    
    
    
    #keep only if longer than 10 frames
    tag_ = "yolo-c45-trim10" 
    pl2 = []
    new_id = 1
    for p in pl:
        if len(p.f_vec) > 10:
            pl2.append(p)
            pl2[-1].pID = new_id
            new_id += 1
    iscat.save_particle_list_video(r8, pl2, constants, tag_)
    generate_output(pl2, constants, tag_)
    
    #save_all_particle_images(pl2, constants, tag_)
    
    #iscat.save_particle_list_video(r8, pl, constants, tag_)
    #generate_output(pl2, constants, tag_)
    
    
    #iscat.save_si_video(r8, pl2, constants, "-si-")
    
    # #keep only if maxmimum confidence is above 0.25 or whatever
    # tag_ = "std01conf30"
    # pl3 = []
    # new_id = 1
    # for p in pl2:
    #     if np.max(p.stddev_vec) < 0.01 and np.max(p.conf_vec) > 0.30:          # this mean we need to be at least 30% sure its a particle in order to accept a small one
    #         pl3.append(p)
    #         pl3[-1].pID = new_id
    #         new_id += 1
    #     if np.max(p.stddev_vec) >= 0.01 and np.max(p.conf_vec) > 0.12:         # larger particles only need to be 12% sure
    #         pl3.append(p)
    #         pl3[-1].pID = new_id
    #         new_id += 1
    # iscat.save_particle_list_video(r8, pl3, constants, tag_)
    # iscat.save_bw_video(r8, constants, "ratiometric", print_frame_nums=False)
    # generate_output(pl3, constants, tag_)
    # save_all_particle_images(pl3, constants, tag_)
    
    
    #keep only if maxmimum confidence is above 0.10 or whatever
    # tag_ = "conf12"
    # pl3 = []
    # new_id = 1
    
    # for p in pl2:
    #     if np.max(p.conf_vec) > 0.12:
    #         pl3.append(p)
    #         pl3[-1].pID = new_id
    #         new_id += 1
    # iscat.save_particle_list_video(r8, pl3, constants, tag_)
    # iscat.save_bw_video(r8, constants, "ratiometric", print_frame_nums=False)
    # generate_output(pl3, constants, tag_)
    # save_all_particle_images(pl3, constants, tag_)
    
    
    # tag_ = "sd-025 cut"
    # pl_new = []
    # new_id = 1
    # for p in pl3:
    #     if np.max(p.stddev_vec) > 0.025:
    #        pl_new.append(p)
    #        pl_new[-1].pID = new_id
    #        new_id += 1
    # #iscat.save_particle_list_video(r8, pl3, constants, tag_)
    # generate_output(pl_new, constants, tag_)
            
        
        
    
    
    
    
    ###########################################################################
    et = time.time()
    print("\n", name)
    print("\n\nFINSIHED... Processed ", os.path.getsize(binfile), " bytes in ", ((et-st)/60), " minutes\n\n")
        
 

#%%    



#%%

# binfile = r"C:/Users/Matt/Desktop/PS ITO DATA wait 1 minute/25 nm/2023-03-23-25nm new ITO 150mW/VIDEOS/2023-03-23_15-09-20_raw_12_256_200.bin"
# sample_name = "test"
# generate_exp_data(binfile, sample_name)


#         ____          __         __               
#        / __ ) ____ _ / /_ _____ / /_              
#       / __  |/ __ `// __// ___// __ \             
#      / /_/ // /_/ // /_ / /__ / / / /             
#     /_____/ \__,_/ \__/ \___//_/ /_/              
#         ____                                      
#        / __ \ _____ ____   _____ ___   _____ _____
#       / /_/ // ___// __ \ / ___// _ \ / ___// ___/
#      / ____// /   / /_/ // /__ /  __/(__  )(__  ) 
#     /_/    /_/    \____/ \___/ \___//____//____/  
#         ______        __     __                   
#        / ____/____   / /____/ /___   _____        
#       / /_   / __ \ / // __  // _ \ / ___/        
#      / __/  / /_/ // // /_/ //  __// /            
#     /_/     \____//_/ \__,_/ \___//_/        

''' PROCESS A WHOLE FOLDER OF FOLDERS OF .BINS '''
def batch_process_exp_data(root_dir):
    
    binfiles = []
    folders = []
    sample_names = []

    for f in os.listdir(root_dir):
        
        sample_name = f[11:]
        folder = os.path.join(root_dir, f, 'VIDEOS')
        for filename in os.listdir(folder):
            if filename.endswith('.bin'):
                binfile = os.path.join(folder, filename)
                
                binfiles.append(binfile)
                folders.append(folder)
                sample_names.append(sample_name)
                
                print(binfile, "\n", sample_name, "\n\n")
                
                generate_exp_data(binfile, sample_name)
                
    return binfiles, folders, sample_names







'''#######################################################################
  _________     _______  ______   ______ ____  _      _____  ______ _____  
 |__   __\ \   / /  __ \|  ____| |  ____/ __ \| |    |  __ \|  ____|  __ \ 
    | |   \ \_/ /| |__) | |__    | |__ | |  | | |    | |  | | |__  | |__) |
    | |    \   / |  ___/|  __|   |  __|| |  | | |    | |  | |  __| |  _  / 
    | |     | |  | |    | |____  | |   | |__| | |____| |__| | |____| | \ \ 
  _ |_|     |_|  |_| __ |______| |_|  _ \____/|______|_____/|______|_|  \_\
 | \ | |   /\   |  \/  |  ____| | |  | |  ____|  __ \|  ____|              
 |  \| |  /  \  | \  / | |__    | |__| | |__  | |__) | |__                 
 | . ` | / /\ \ | |\/| |  __|   |  __  |  __| |  _  /|  __|                
 | |\  |/ ____ \| |  | | |____  | |  | | |____| | \ \| |____               
 |_| \_/_/    \_\_|  |_|______| |_|  |_|______|_|  \_\______|     
 
 '''


root_dir = r"D:/ACS NANO TESTS/7-5 large particle centering tests"
root_dir = r"D:/ACS NANO TESTS/PS - Contrast 3"
root_dir = r"D:/Carraugh/Aug 29"
root_dir = r"D:/_DATA FOLDER_/Teresa"
root_dir = r"D:/_DATA FOLDER_/Haoxin"
root_dir = r"D:/_DATA FOLDER_/Haoxin 2"
root_dir = r"D:/_DATA FOLDER_/extra 0.3 nM data"
root_dir = r"D:/_DATA FOLDER_/haoxin 3"
strt = time.time()

a, b, c = batch_process_exp_data(root_dir)

endt = time.time()
print("Total Runtime: ", ((endt-strt)/60), " minutes")





