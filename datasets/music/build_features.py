# Given an array of PCM samples, return a feature matrix
# The feature matrix doesn't need to be upsampled to the sample rate
# However long the matrix is, we assume it matches the length of the WAV
# So you can use any frame_rate (hop_size)
import numpy as np
import librosa
import librosa.onset
def build_dummy_features(filename):
    features = np.ones((1000,1),dtype='float32')
    for i,_ in enumerate(features):
	    features[i,0] = i/1000.0 
    return features

def build_onset_envelope_feature(filename):
	y, sr = librosa.load(filename)
	hop_length=128
	onset_env = librosa.onset.onset_strength(y=y, sr=sr, 
		aggregate=np.median, hop_length=hop_length, fmax=8000)
	# normalize the onset_env
	onset_env = (onset_env - np.mean(onset_env))/np.std(onset_env)

	num_frames = len(onset_env)
	feature_matrix = np.ones((num_frames,1),dtype='float32')
	feature_matrix[:,0] = onset_env
	return feature_matrix

