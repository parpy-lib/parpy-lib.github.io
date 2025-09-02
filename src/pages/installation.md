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

Below, we provide more detailed installation instructions, including:
- The pre-installation steps required to install ParPy and to enable a GPU backend ([Pre-installation steps](#pre-installation-steps)).
- Detailed installation instructions for ParPy, with alternative Conda environments for running benchmarks ([Installing ParPy](#installing-parpy)).
- How to run the test suite after installing ParPy ([Running tests](#running-tests)).
- Where to find documentation on how to use it, including examples ([More information](#more-information)).

## Pre-installation steps

These installation instructions assume [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) has been installed and set up on your system. We provide Conda environments that include all dependencies needed to enable ParPy.

To enable executing the generated GPU code, you may need to perform additional setup steps, depending on the target backend.

:::info
When using a system lacking the required hardware to run either backend, you can still install ParPy. However, in this case, ParPy can only be used to generate the low-level code (as a string) without the ability to execute it.
:::

### CUDA

CUDA requires an NVIDIA GPU with CUDA-compatible GPU drivers installed. To install the CUDA toolkit and CUDA drivers, follow the instructions [here](https://developer.nvidia.com/cuda-toolkit-archive).

Our provided Conda environment assumes drivers supporting at least CUDA 12.2 are installed. This is required to be able to run all benchmarks.

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

:::info Install Environment
<Tabs>
<TabItem value="parpy-install-minimal" label="Minimal" default>

Install the minimal Conda environment, including dependencies required to install and run ParPy, by running
```bash
conda env create -f benchmarks/minimal-env.yml
```

</TabItem>
<TabItem value="parpy-install-cuda" label="CUDA">

Install the Conda environment including dependencies required to run all CUDA benchmarks by running
```bash
conda env create -f benchmarks/cuda-env.yml
```

</TabItem>
<TabItem value="parpy-install-metal" label="Metal">

Install the Conda environment including dependencies required to run all Metal benchmarks by running
```bash
conda env create -f benchmarks/metal-env.yml
```

</TabItem>
</Tabs>
:::

After setting up the environment, run
```bash
conda activate parpy-env
```
to activate the Conda environment. To exit the environment, run `conda deactivate`. Within the environment, install ParPy from the root directory by running
```bash
pip install .
```

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

## More information

See the [Documentation](/docs) for examples of how to use ParPy and documentation of the exposed ParPy API. In particular, the tutorial on [printing generated code](/docs/tutorials/print-code) provides an example of how to use ParPy without having to set up any backend.
