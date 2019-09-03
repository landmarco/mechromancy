import subprocess
import sys
import glob

DIR = "."
fs = glob.glob(DIR+"/*.flac")
for f in fs:
    size = float(subprocess.check_output('ffprobe -i "{}/{}" -show_entries format=duration -v quiet -of csv="p=0"'.format(DIR, f), shell=True))
    if size != 3.762563:
        print f
        print size