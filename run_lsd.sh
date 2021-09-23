cd /home/yakovdan/repos
git clone https://github.com/yakovdan/Datasets.git
cd /home/yakovdan/Background-Subtraction/
python3 ./inexact_alm_lsd.py --input /home/yakovdan/repos/Datasets/... --output ./output/smallscale/ --frame_start 0 --frame_end 862 --downscale 4
python3 ./inexact_alm_lsd.py --input /home/yakovdan/repos/Datasets/... --output ./output/fullscale/ --frame_start 0 --frame_end 862 --downscale 1