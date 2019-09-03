SCRIPTPATH=$( cd "$(dirname "$0")" ; pwd -P )
echo "Preprocessing"
python drum-preprocess.py "$SCRIPTPATH"
echo "Done!"

echo "Writing datasets"
python _drum2npy.py
echo "Done!"
