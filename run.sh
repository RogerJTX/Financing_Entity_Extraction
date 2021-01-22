set -e
if [ -d "output" ];then
    rm -rf output
fi
mkdir output
touch output/config.json
python train.py

