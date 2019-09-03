import numpy as np
import sys, os, subprocess, scikits.audiolab, random, time, glob, math


from build_features import * 

PWD = os.getcwd()
print 'PWD is', PWD
#store dataset name
DATASET_NAME = str(sys.argv[1])
DOWNLOAD_DIR = str(sys.argv[2])
print 'dl_dir is set to', DOWNLOAD_DIR
#create the 
print "creating directory for", DATASET_NAME
DATASET_DIR = os.path.join(PWD, DATASET_NAME)
os.makedirs(DATASET_DIR)
#move samples from directory to use dataset name
print "moving samples"
types = {'wav', "mp3"}
for t in types:
    os.system('mv {}/*.{} {}/'.format(DOWNLOAD_DIR, t, DATASET_DIR))
#run proprocess
print "preprocessing"
OUTPUT_DIR=os.path.join(DATASET_DIR, "parts")
os.makedirs(OUTPUT_DIR)
# Step 1: write all filenames to a list
with open(os.path.join(DATASET_DIR, 'preprocess_file_list.txt'), 'w') as f:
    for dirpath, dirnames, filenames in os.walk(DATASET_DIR):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith("mp3"):
                f.write("file '" + dirpath + '/'+ filename + "'\n")

# Step 2: concatenate everything into one massive wav file
print "concatenate all files"
os.system('pwd')
os.system("ffmpeg -f concat -safe 0 -i {}/preprocess_file_list.txt {}/preprocess_all_audio.wav".format(DATASET_DIR, OUTPUT_DIR))
audio = "preprocess_all_audio.wav"
print "get length"
# # get the length of the resulting file
length = float(subprocess.check_output('ffprobe -i {}/{} -show_entries format=duration -v quiet -of csv="p=0"'.format(OUTPUT_DIR, audio), shell=True))
print length, "DURATION"
print "print big file into chunks"
# # Step 3: split the big file into 8-second chunks
# overlapping 3 times per 8 seconds
'''
for i in xrange(int((length//8)*3)-1):
    time = (i * 8 )/ 3
    os.system('ffmpeg -ss {} -t 8 -i {}/preprocess_all_audio.wav -ac 1 -ab 16k -ar 16000 {}/p{}.flac'.format(time, OUTPUT_DIR, OUTPUT_DIR, i))
'''

# size in seconds of each chunk
size = 8
# number of chunks
num_chunks = 3200

# cj (conditioning) generate the feature matrix for the entire dataset WAV
features = build_onset_envelope_feature("{}/preprocess_all_audio.wav".format(OUTPUT_DIR))
# frame_rate is the number of feature frames per second
# calcualte it by comparing length of features to length of audio 
# don't confuse feature_frames for the SampleRNN frames
total_num_frames = features.shape[0]
num_features = features.shape[1]
frame_rate = len(features)/float(length)
# number of frames per chunk of audio 
frames_per_chunk = int(math.floor((size)*frame_rate))
# a matrix of chunks x frames x features
feature_matrix = np.zeros((num_chunks, frames_per_chunk, num_features), dtype='float32')



for i in xrange(0, num_chunks):
    time = i * ((length-size)/float(num_chunks))

    # build the feature_matrix
    # it's the feature timesliced according to the start and end times of the chunk
    start_frame = int(math.floor((time)*frame_rate))
    end_frame = start_frame + frames_per_chunk
    if(len(features)<=end_frame): 
        end_frame = len(features)-1
    # print "start_frame", start_frame
    # print "end_frame", end_frame
    # print "features[start:end].shape", features[start_frame:end_frame].shape
    # print "len(features)", len(features)
    # print "time", time 
    # print "frames_per_chunk", frames_per_chunk
    # print "frame_rate", frame_rate
    # print "total_num_frames", total_num_frames
    # print "num_features", num_features
    feature_matrix[i] = features[start_frame:end_frame]

    os.system('ffmpeg -ss {} -t 8 -i {}/preprocess_all_audio.wav -ac 1 -ab 16k -ar 16000 {}/p{}.flac'.format(time, OUTPUT_DIR, OUTPUT_DIR, i))
print "clean up"



# # Step 4: clean up temp files
os.system('rm {}/preprocess_all_audio.wav'.format(OUTPUT_DIR))
os.system('rm {}/preprocess_file_list.txt'.format(DATASET_DIR))
print 'save as .npy'
__RAND_SEED = 123
def __fixed_shuffle(inp_list):
    if isinstance(inp_list, list):
        random.seed(__RAND_SEED)
        random.shuffle(inp_list)
        return
    #import collections
    #if isinstance(inp_list, (collections.Sequence)):
    if isinstance(inp_list, numpy.ndarray):
        numpy.random.seed(__RAND_SEED)
        numpy.random.shuffle(inp_list)
        return
    # destructive operations; in place; no need to return
    raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

paths = sorted(glob.glob(OUTPUT_DIR+"/*.flac"))
__fixed_shuffle(paths)





# CJ (conditioning)
# For conditioning, the np_arr should be structured as follows
# np_arr[0] are the PCM samples as usual
# np_arr[1] are the feature vectors 

# Turn the FLACs into PCM samples
samples = np.array([(scikits.audiolab.flacread(p)[0]).astype('float16') for p in paths])
print samples.shape

print feature_matrix.shape 


# 88/6/6 split
length = samples.shape[0]
train_size = int(np.floor(length * .88)) # train
test_size = int(np.floor(length * .06)) # test

np.save(os.path.join(DATASET_DIR,'all_music.npy'), samples)
np.save(os.path.join(DATASET_DIR,'music_train.npy'), samples[:train_size])
np.save(os.path.join(DATASET_DIR,'music_valid.npy'), samples[train_size:train_size + test_size])
np.save(os.path.join(DATASET_DIR,'music_test.npy'), samples[train_size + test_size:])

np.save(os.path.join(DATASET_DIR,'all_features.npy'), feature_matrix)
np.save(os.path.join(DATASET_DIR,'features_train.npy'), feature_matrix[:train_size])
np.save(os.path.join(DATASET_DIR,'features_valid.npy'), feature_matrix[train_size:train_size + test_size])
np.save(os.path.join(DATASET_DIR,'features_test.npy'), feature_matrix[train_size + test_size:])

#pass dataset name through two_tier.py || three_tier.py to datasets.py