import os
from time import time
import scipy.io.wavfile
import glob
import sys
import numpy
import pickle
import theano
import theano.tensor as T

tag = sys.argv[1]
name = glob.glob("../results*/" + tag + "/args.pkl")[0]
params = pickle.load(open(name, "r"))
print params
info = {}
for p in xrange(1,len(params),2):
    if p+1 < len(params):
        info[params[p][2:]] = params[p+1]
print info
#exit()

Q_TYPE = info["q_type"]
Q_LEVELS = int(info["q_levels"])
N_RNN = int(info["n_rnn"])
DIM = int(info["dim"])
FRAME_SIZE = int(info["frame_size"])


#{'dim': '1024', 'q_type': 'linear', 'learn_h0': 'True', 'weight_norm': 'True', 'q_levels': '256', 'skip_conn': 'False', 'batch_size': '128', 'n_frames': '64', 'emb_size': '256', 'exp': 'KURT2x4', 'frame_size': '16', 'which_set': 'KURT', 'rnn_type': 'GRU', 'n_rnn': '4'}

###grab this stuff
#args
#Q_TYPE
#Q_TEVELS
#N_RNN
#DIM
#FRAME_SIZE

BITRATE = 16000
N_SEQS = 20  # Number of samples to generate every time monitoring.
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
H0_MULT = 1

RESULTS_DIR = 'results_2t'
RESULTS_DIR = name.split("/")[1]
print RESULTS_DIR

FOLDER_PREFIX = os.path.join(RESULTS_DIR, tag)
### Create directories ###
#   FOLDER_PREFIX: root, contains:
#       log.txt, __note.txt, train_log.pkl, train_log.png [, model_settings.txt]
#   FOLDER_PREFIX/samples: keeps all checkpoint samples as wav
SAMPLES_PATH = os.path.join(FOLDER_PREFIX, 'samples')

print SAMPLES_PATH
# Uniform [-0.5, 0.5) for half of initial state for generated samples
# to study the behaviour of the model and also to introduce some diversity
# to samples in a simple way. [it's disabled for now]
sequences = T.imatrix('sequences')
h0        = T.tensor3('h0')
reset     = T.iscalar('reset')
mask      = T.matrix('mask')
fixed_rand_h0 = numpy.random.rand(N_SEQS//2, N_RNN, H0_MULT*DIM)
fixed_rand_h0 -= 0.5
fixed_rand_h0 = fixed_rand_h0.astype('float32')

def generate_and_save_samples():
    # Sampling at frame level
    frame_level_generate_fn = theano.function(
        [sequences, h0, reset],
        frame_level_rnn(sequences, h0, reset),
        on_unused_input='warn'
    )
    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(
                    os.path.join(SAMPLES_PATH, name+'.wav'),
                    BITRATE,
                    data)

    total_time = time()
    # Generate N_SEQS' sample files, each 5 seconds long
    N_SECS = 5
    LENGTH = N_SECS*BITRATE

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples[:, :FRAME_SIZE] = Q_ZERO

    # First half zero, others fixed random at each checkpoint
    h0 = numpy.zeros(
            (N_SEQS-fixed_rand_h0.shape[0], N_RNN, H0_MULT*DIM),
            dtype='float32'
    )
    h0 = numpy.concatenate((h0, fixed_rand_h0), axis=0)
    frame_level_outputs = None

    for t in xrange(FRAME_SIZE, LENGTH):

        if t % FRAME_SIZE == 0:
            frame_level_outputs, h0 = frame_level_generate_fn(
                samples[:, t-FRAME_SIZE:t],
                h0,
                #numpy.full((N_SEQS, ), (t == FRAME_SIZE), dtype='int32'),
                numpy.int32(t == FRAME_SIZE)
            )

        samples[:, t] = sample_level_generate_fn(
            frame_level_outputs[:, t % FRAME_SIZE],
            samples[:, t-FRAME_SIZE:t],
        )

    total_time = time() - total_time
    log = "{} samples of {} seconds length generated in {} seconds."
    log = log.format(N_SEQS, N_SECS, total_time)
    print log,

    for i in xrange(N_SEQS):
        samp = samples[i]
        if Q_TYPE == 'mu-law':
            from datasets.dataset import mu2linear
            samp = mu2linear(samp)
        elif Q_TYPE == 'a-law':
            raise NotImplementedError('a-law is not implemented')
        write_audio_file("sample_{}_{}".format(tag, i), samp)

generate_and_save_samples()