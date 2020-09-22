
cd /home/sandbox/Git_projects/open_AR_Sandbox

conda init

conda activate sandbox

git fetch origin

git checkout seismic

git pull

git lfs pull

sudo cp /home/sandbox/Git_projects/open_AR_Sandbox/notebooks/tutorials/10_SeismicModule/open_jupyter.sh /home/sandbox/

jupyter notebook --no-browser
