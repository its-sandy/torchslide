# torchslide
A custom drop in PyTorch module for SLIDE layer introduced in [SLIDE : IN DEFENSE OF SMART ALGORITHMS OVER HARDWARE ACCELERATION FOR LARGE-SCALE DEEP LEARNING SYSTEMS](https://arxiv.org/pdf/1903.03129.pdf)

## Running Experiments
Go to `./SLIDE/cpp` and run `python setup.py install` to build the Torch cpp extension.
Refer to `./experiments.py` for info on using the different available modes of sparse multiply operations.
