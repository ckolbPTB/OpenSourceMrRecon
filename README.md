# Open source image reconstruction for MRI

With this container you can start to reconstruct MR data straight away - it has [Gadgetron](https://github.com/gadgetron/gadgetron), [BART](http://mrirecon.github.io/bart/), [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) and [SIRF](https://github.com/SyneRBI/SIRF) and all their dependencies installed. 

## Build docker image
In order to build the docker image, clone this repository
```
git clone https://github.com/ckolbPTB/OpenSourceMrRecon.git
```

Change into the directory
```
cd OpenSourceMrRecon
```

and build docker image
```
docker build -t os_recon .
```

Here we are tagging the image with *os_recon*. Feel free to use any other tag, but make sure you select the correct image when starting the container.  

The build process will take some time (on a normal laptop around 1.5 hours) and the final image will be around 9GB in size.

## Start container
Once the build is successfully finished, start the container in interactive mode
```
docker run -it os_recon
```

Enable the conda environment with all dependencies installed
```
source /opt/conda/bin/activate RecoEnv
```

Now you are ready to go!

## Reproduce figure from paper
If you want to reproduce the figure showing the different reconstructions of the cardiac spin echo data, then you only need to mount a folder when starting the container where the figure can be saved. 
```
docker run -it -v full_path_to_your_folder:/recon_output os_recon
```

Once the container has started, make sure to enable the conda environment with all dependencies installed
```
source /opt/conda/bin/activate RecoEnv
```

Start a gadgetron server (and hit enter afterwards to return to the command line)
```
gadgetron &
```

Run the python script with all reconstructions
```
python example_code/run_open_source_recon.py
```

You will see lots of output in the terminal but that's all fine. Once the script has finished you should find the figure *open_source_recon.png* in *full_path_to_your_folder*.

