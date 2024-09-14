# Dixon-based B0 self-navigation in radial stack-of-stars multi-echo gradient echo imaging

Julia implementation of the Dixon B0 self-navigator based on the publication:

*Jonathan Stelter, Kilian Weiss, Mingming Wu, Johannes Raspe, Philipp Braun, Christoph Zöllner, Dimitrios C. Karampinos; Dixon‐based B0 self‐navigation in radial stack‐of‐stars multi‐echo gradient echo imaging, Magnetic Resonance in Medicine, Magnetic Resonance in Medicine, DOI: 10.1002/mrm.30261, https://doi.org/10.1002/mrm.30261*

The implementation of the B0 self-navigator can be found in src/Corrections/DeltaB0Correction.jl. 
Scripts for reproducing the simulations (with the exception of the commercial XCAT phantom) and phantom reconstruction are shared. The reconstruction was performed in Julia and the post-processing and evaluation in Python.

## 🚀 Setup

### Requirements

- Julia 1.9 (system-wide installation is recommened)
- Anaconda/mamba environment with Python 3.10
- NVIDIA GPU recommended

### Installing

1. **Create a new mamba environment:**

    ```shell
    mamba env create --name b0nav --file environment_nocuda.yml
    mamba activate b0nav
    which python
    ```

2. **Open Julia and instantiate/precompile new Julia environment:**

    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ENV["PYTHON"] = "/path/to/envs/b0nav/bin/python"
    Pkg.build("PyCall")
    ```

3. **Run processing script directly from the shell:**

    Replace `n_threads` with the number of threads you wish to use.

    ```shell
    julia -t n_threads -i scripts/2_1_recon_simulation_geo.jl
    ```

## Data
Raw data for the simulation/phantom (>12GB) experiment are stored at the [OneDrive folder](https://1drv.ms/f/s!AsJuTLjw5_n7ls9GYDpokf-_jylcYw?e=1rL1If).

## Authors and acknowledgment
* Jonathan Stelter - [Body Magnetic Resonance Research Group, TUM](http://bmrr.de)

**Water-fat separation: https://github.com/BMRRgroup/fieldmapping-hmrGC**

## License
This project is licensed as given in the LICENSE file. However, used submodules / projects may be licensed differently. Please see the respective licenses.