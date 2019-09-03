import os, sys
import subprocess
# requires sox, ffmpeg, and ffprobe command line tools

RAW_DATA_DIR=str(sys.argv[1])
TEMP_DIR=os.path.join(RAW_DATA_DIR, "temp")
FR_DIR=os.path.join(RAW_DATA_DIR, "fr-parts")
BR_DIR=os.path.join(RAW_DATA_DIR, "br-parts")
F_DIR=os.path.join(RAW_DATA_DIR, "f-parts")
B_DIR=os.path.join(RAW_DATA_DIR, "b-parts")
SAMPLE_RATE = 16000
os.makedirs(TEMP_DIR)
os.makedirs(FR_DIR)
os.makedirs(BR_DIR)
os.makedirs(F_DIR)
os.makedirs(B_DIR)

def createParts():
    def renderFlacs(fr, br, f, b):
        os.system('ffmpeg -i {}/{}_temp.wav -ac 1 -ab 16k -ar {} {}/p{}.flac'.format(TEMP_DIR, fr, SAMPLE_RATE, FR_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_temp.wav -ac 1 -ab 16k -ar {} {}/p{}.flac'.format(TEMP_DIR, br, SAMPLE_RATE, BR_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_temp.wav -ac 1 -ab 16k -ar {} {}/p{}.flac'.format(TEMP_DIR, f, SAMPLE_RATE, F_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_temp.wav -ac 1 -ab 16k -ar {} {}/p{}.flac'.format(TEMP_DIR, b, SAMPLE_RATE, B_DIR, i))#convert part to flac
        #pitch down
        os.system('ffmpeg -i {}/{}_down.wav -ac 1 -ab 16k -ar {} {}/p{}d.flac'.format(TEMP_DIR, fr, SAMPLE_RATE, FR_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_down.wav -ac 1 -ab 16k -ar {} {}/p{}d.flac'.format(TEMP_DIR, br, SAMPLE_RATE, BR_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_down.wav -ac 1 -ab 16k -ar {} {}/p{}d.flac'.format(TEMP_DIR, f, SAMPLE_RATE, F_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_down.wav -ac 1 -ab 16k -ar {} {}/p{}d.flac'.format(TEMP_DIR, b, SAMPLE_RATE, B_DIR, i))#convert part to flac       
        #pitch up
        os.system('ffmpeg -i {}/{}_up.wav -ac 1 -ab 16k -ar {} {}/p{}u.flac'.format(TEMP_DIR, fr, SAMPLE_RATE, FR_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_up.wav -ac 1 -ab 16k -ar {} {}/p{}u.flac'.format(TEMP_DIR, br, SAMPLE_RATE, BR_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_up.wav -ac 1 -ab 16k -ar {} {}/p{}u.flac'.format(TEMP_DIR, f, SAMPLE_RATE, F_DIR, i))#convert part to flac
        os.system('ffmpeg -i {}/{}_up.wav -ac 1 -ab 16k -ar {} {}/p{}u.flac'.format(TEMP_DIR, b, SAMPLE_RATE, B_DIR, i))#convert part to flac
    #initial preparation
    os.system('ffmpeg -i "{}" -ac 1 -ab 16k -ar {} {}/this_temp.wav'.format(full_name, SAMPLE_RATE, TEMP_DIR)) #resample this file as mono 16000smpls/s
    this_length = float(subprocess.check_output('ffprobe -i {}/this_temp.wav -show_entries format=duration -v quiet -of csv="p=0"'.format(TEMP_DIR), shell=True)) #check length of resampled audio
    print full_name, ':', this_length, 'DURATION'
    pad_length =  longest_length - this_length
    os.system('sox {}/this_temp.wav {}/r_temp.wav reverse'.format(TEMP_DIR, TEMP_DIR)) # reverse file
    if pad_length > 0.: # every audiofile except the largest
        #create temp files
        os.system('ffmpeg -f lavfi -i anullsrc=channel_layout=mono:sample_rate={} -t {} {}/anullsrc_temp.wav'.format(SAMPLE_RATE, pad_length, TEMP_DIR)) #create anullsrc_temp.wav zero-pad
        os.system('sox {}/anullsrc_temp.wav {}/r_temp.wav {}/fr_temp.wav'.format(TEMP_DIR, TEMP_DIR, TEMP_DIR)) #FR
        os.system('sox {}/r_temp.wav {}/anullsrc_temp.wav {}/br_temp.wav'.format(TEMP_DIR, TEMP_DIR, TEMP_DIR)) #BR
        os.system('sox {}/anullsrc_temp.wav {}/this_temp.wav {}/f_temp.wav'.format(TEMP_DIR, TEMP_DIR, TEMP_DIR)) #F
        os.system('sox {}/this_temp.wav {}/anullsrc_temp.wav {}/b_temp.wav'.format(TEMP_DIR, TEMP_DIR, TEMP_DIR)) #B
        # extend the data set by copying and repitching each sample up+down 1 semitone
        os.system('sox {}/fr_temp.wav {}/fr_down.wav pitch -100'.format(TEMP_DIR, TEMP_DIR))#FR down
        os.system('sox {}/br_temp.wav {}/br_down.wav pitch -100'.format(TEMP_DIR, TEMP_DIR))#BR down
        os.system('sox {}/f_temp.wav {}/f_down.wav pitch -100'.format(TEMP_DIR, TEMP_DIR))#F down
        os.system('sox {}/b_temp.wav {}/b_down.wav pitch -100'.format(TEMP_DIR, TEMP_DIR))#B down
        os.system('sox {}/fr_temp.wav {}/fr_up.wav pitch 100'.format(TEMP_DIR, TEMP_DIR))#FR up
        os.system('sox {}/br_temp.wav {}/br_up.wav pitch 100'.format(TEMP_DIR, TEMP_DIR))#BR up
        os.system('sox {}/f_temp.wav {}/f_up.wav pitch 100'.format(TEMP_DIR, TEMP_DIR))#F up
        os.system('sox {}/b_temp.wav {}/b_up.wav pitch 100'.format(TEMP_DIR, TEMP_DIR))#D up
        #final export
        renderFlacs('fr', 'br', 'f', 'b') #render parts
        #clean up temp files
        os.system('rm {}/anullsrc_temp.wav'.format(TEMP_DIR))
        os.system('rm {}/fr_down.wav'.format(TEMP_DIR))
        os.system('rm {}/br_down.wav'.format(TEMP_DIR))
        os.system('rm {}/f_down.wav'.format(TEMP_DIR))
        os.system('rm {}/b_down.wav'.format(TEMP_DIR))
        os.system('rm {}/fr_up.wav'.format(TEMP_DIR))
        os.system('rm {}/br_up.wav'.format(TEMP_DIR))
        os.system('rm {}/f_up.wav'.format(TEMP_DIR))
        os.system('rm {}/b_up.wav'.format(TEMP_DIR))
    else: #longest file
        # extend the data set by copying and repitching each sample up+down 1 semitone
        os.system('sox {}/this_temp.wav {}/r_up.wav pitch 100'.format(TEMP_DIR, TEMP_DIR))# up
        os.system('sox {}/this_temp.wav {}/r_down.wav pitch -100'.format(TEMP_DIR, TEMP_DIR))# down
        os.system('sox {}/r_temp.wav {}/this_up.wav pitch 100'.format(TEMP_DIR, TEMP_DIR))#r up
        os.system('sox {}/r_temp.wav {}/this_down.wav pitch -100'.format(TEMP_DIR, TEMP_DIR))#r down
        # final export
        renderFlacs('r', 'r', 'this', 'this')
        #clean up temp files
        os.system('rm {}/r_up.wav'.format(TEMP_DIR))
        os.system('rm {}/r_down.wav'.format(TEMP_DIR))
        os.system('rm {}/this_up.wav'.format(TEMP_DIR))
        os.system('rm {}/this_down.wav'.format(TEMP_DIR))
    os.system('rm {}/r_temp.wav'.format(TEMP_DIR))
    os.system('rm {}/this_temp.wav'.format(TEMP_DIR))

# Step 1: Find the largest file size in the audio dataset
objects = os.listdir(RAW_DATA_DIR)
sofar = 0
largest = ""
for item in objects:
    if ".wav" in item:
        size = os.path.getsize(item)
        if size > sofar:
                sofar = size
                largest = item

print "Largest file is ", sofar
print largest
os.system('ffmpeg -i "{}" -ac 1 -ab 16k -ar {} {}/longest_temp.wav'.format(largest, SAMPLE_RATE, TEMP_DIR)) #resample the largest file as mono 
longest_length = float(subprocess.check_output('ffprobe -i {}/longest_temp.wav -show_entries format=duration -v quiet -of csv="p=0"'.format(TEMP_DIR), shell=True))
#clean up longest temp wav
os.system('rm {}/longest_temp.wav'.format(TEMP_DIR))

i = 0
for dirpath, dirnames, filenames in os.walk(RAW_DATA_DIR):
    for filename in filenames:
        if filename.endswith(".wav"):
            full_name = dirpath + '/'+ filename # raw audio file
            createParts()
            i += 1
#remove empty temp dir
#os.system('rmdir {}'.format(TEMP_DIR))
