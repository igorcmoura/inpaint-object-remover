# Inpaint Object Remover
This is the implementation of the algorithm described in the paper "Region Filling and Object Removal by
Exemplar-Based Image Inpainting" by A. Criminisi et al.

## Setup
Requires python 3.5 or newer. You can download it [here](https://www.python.org/downloads/).

### Linux and Mac OS X
Inside the project's directory run:
```
pip install -r requirements.txt
```
to install the dependencies.

### Windows
Download NumPy and SciPy from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/), being the number after cp your version of python. (e.g. if you have python 3.5 64 bits download **numpy‑1.13.1+mkl‑cp35‑cp35m‑win_amd64.whl** and **scipy‑0.19.1‑cp35‑cp35m‑win_amd64.whl**)

Then run `pip install <path to downloaded file>` for each file.

After this, inside the project's directory, run:
```
pip install -r requirements.txt
```
to install the other dependencies.

## How to use
Inside the project's directory run:
```
python inpainter <path to image> <path to mask>
```
You can also pass the `--plot-progress` argument to watch the image being created. For example, use:
```
python inpainter resources/image1.jpg resources/mask1.jpg --plot-progress
```
to process image1 inside resources folder using mask1 while printing each result. I provided some images from the paper inside the resources folder.

Use `python inpainter -h` to show the available arguments.
