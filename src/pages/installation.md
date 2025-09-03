import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Installation

This page covers how to install ParPy and configure its various backends to run the test suite.

## Quick install

To install ParPy, run the following commands:
```bash
git clone https://github.com/parpy-lib/ParPy.git
cd ParPy
conda env create -f benchmarks/minimal-env.yml
conda activate parpy-env
pip install .
```

To verify that the installation works, run the compiler tests as shown below. Note that the latter command takes a few minutes to run.
```bash
cargo test
pytest
```

Next up is the tutorial on [Basic parallelization](/docs/tutorials/basic-parallelization) in ParPy for a brief introduction. Below, we provide detailed installation instructions, including:
- The pre-installation steps required to install ParPy and to enable a GPU backend ([Pre-installation steps](#pre-installation-steps)).
- Detailed installation instructions for ParPy, with alternative Conda environments for running benchmarks ([Installing ParPy](#installing-parpy)).
- More details on the test suite and what the commands provided above do ([Running tests](#running-tests)).

## Pre-installation steps

These installation instructions assume [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) has been installed and set up on your system. We provide Conda environments that include all dependencies needed to enable ParPy.

To enable executing the generated GPU code, you may need to perform additional setup steps, depending on the target backend.

:::info
When using a system lacking the required hardware to run either backend, you can still install ParPy. However, in this case, ParPy can only be used to generate the low-level code (as a string) without the ability to execute it.
:::

### CUDA

CUDA requires an NVIDIA GPU with CUDA-compatible GPU drivers installed. To install the CUDA toolkit and CUDA drivers, follow the instructions [here](https://developer.nvidia.com/cuda-toolkit-archive).

Our provided Conda environment assumes drivers supporting at least CUDA 12.2 are installed. This is required to be able to run all benchmarks. Further, ParPy assumes the `nvcc` command is included in the PATH.

### Metal

The Metal backend requires Apple Silicon hardware running on macOS. In addition, the XCode command-line tools must be installed:
```bash
xcode-select install
```

In addition, the [Metal C++](https://developer.apple.com/metal/cpp/) header library must be installed for the correct macOS version. After downloading the files, the `METAL_CPP_HEADER_PATH` environment variable must be set to the path of the `metal-cpp` directory. For instance, if the `metal-cpp` directory is located at `/Users/user/Documents`, you can set the environment variable by running:
```bash
export METAL_CPP_HEADER_PATH=/Users/user/Documents/metal-cpp
```

:::info
Run the `sw_vers --productVersion` command to show the installed macOS version.
:::

## Installing ParPy

Clone the ParPy repository to a suitable location on your system and enter the root directory:
```bash
git clone https://github.com/parpy-lib/ParPy.git
cd ParPy
```

We provide three Conda environments in the `benchmarks` directory: a minimal installation, including the minimum requirements to be able to install the ParPy compiler and run tests, and environments for the CUDA and Metal backends including dependencies needed to run benchmarks. The environments include supported versions of Python and Rust.

:::info Set up environment and install ParPy
<Tabs>
<TabItem value="parpy-install-minimal" label="Minimal" default>

Install and activate the minimal Conda environment, including dependencies required to install and run ParPy:
```bash
conda env create -f benchmarks/minimal-env.yml
conda activate parpy-env
pip install .
```

</TabItem>
<TabItem value="parpy-install-cuda" label="CUDA">

Install and activate a Conda environment including dependencies required to run all CUDA benchmarks:
```bash
conda env create -f benchmarks/cuda-env.yml
conda activate cuda-parpy-env
pip install .
```

</TabItem>
<TabItem value="parpy-install-metal" label="Metal">

Install and activate a Conda environment including dependencies required to run all Metal benchmarks:
```bash
conda env create -f benchmarks/metal-env.yml
conda activate metal-parpy-env
pip install .
```

</TabItem>
</Tabs>
:::

To exit the Conda environment, run `conda deactivate`.

## Running Tests

ParPy includes both unit tests, which test small components of the native Rust compiler, and integration tests that use the ParPy library. From the root of the ParPy repository, given that the ParPy package has been installed, we run the unit tests as
```bash
cargo test
```

The integration tests, which are found in the `test` directory, are run by using
```bash
pytest
```

:::info Skipped Tests

Many integration tests will be skipped regardless of which backend is enabled. This is because many integration tests execute the generated code from the ParPy compiler.

:::

:::info Metal Backend Warnings

The integration tests will produce a warning if Metal appears to be available on the system, but it is not enabled. This happens when the `METAL_CPP_HEADER_PATH` environment variable has not been set.

:::
