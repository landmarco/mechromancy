import subprocess
import sys
import glob

DIR = "."
fs = glob.glob(DIR+"/*.wav")
t = 0
print 'counting...'
for f in fs:
    size = float(subprocess.check_output('ffprobe -i "{}/{}" -show_entries format=duration -v quiet -of csv="p=0"'.format(DIR, f), shell=True))
    t = t + size   
print t, ' seconds'