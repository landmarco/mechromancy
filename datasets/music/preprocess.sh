# Requires 2GB of free disk space at most.
SCRIPTPATH=$( cd "$(dirname "$0")" ; pwd -P )
echo "Converting from OGG to 16Khz, 16bit mono-channel WAV"
# Next line with & executes in a forked shell in the background. That's parallel and not recommended.
# Remove if causing problem
#for file in "$DL_PATH"*_64kb.mp3; do ffmpeg -i "$file" -ar 16000 -ac 1 "$DL_PATH""`basename "$file" _64kb.mp3`.wav" & done 
for file in "$SCRIPTPATH"*.ogg; do
	ffmpeg -i "$file" -ar 16000 -ac 1 "$SCRIPTPATH""`basename "$file" .ogg`.wav"
done 
echo "Cleaning up"
rm "$SCRIPTPATH"*.ogg

echo "Preprocessing"
python preprocess.py "$SCRIPTPATH"
echo "Done!"

echo "Writing datasets"
python _2npy.py
echo "Done!"
