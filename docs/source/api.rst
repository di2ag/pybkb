API
===

Notes on environment (not formatted for Sphinx)
taken from https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry


Example

environment.yml:

name: my_project_env
channels:
  - pytorch
  - conda-forge
  # We want to have a reproducible setup, so we don't want default channels,
  # which may be different for different users. All required channels should
  # be listed explicitly here.
  - nodefaults
dependencies:
  - python=3.10.*  # or don't specify the version and use the latest stable Python
  - mamba
  - pip  # pip must be mentioned explicitly, or conda-lock will fail
  - poetry=1.*  # or 1.1.*, or no version at all -- as you want
  - tensorflow=2.8.0
  - pytorch::pytorch=1.11.0
  - pytorch::torchaudio=0.11.0
  - pytorch::torchvision=0.12.0

# Non-standard section listing target platforms for conda-lock:
platforms:
  - linux-64

virtual-packages.yml (may be used e.g. when we want conda-lock to generate CUDA-enabled lock files even on platforms without CUDA):

subdirs:
  linux-64:
    packages:
      __cuda: 11.5

First-time setup

You can avoid playing with the bootstrap env and simplify the example below if you have conda-lock, mamba and poetry already installed outside your target environment.

# Create a bootstrap env
conda create -p /tmp/bootstrap -c conda-forge mamba conda-lock poetry='1.*'
conda activate /tmp/bootstrap

# Create Conda lock file(s) from environment.yml
conda-lock -k explicit --conda mamba
# Set up Poetry
poetry init --python=~3.10  # version spec should match the one from environment.yml
# Fix package versions installed by Conda to prevent upgrades
poetry add --lock tensorflow=2.8.0 torch=1.11.0 torchaudio=0.11.0 torchvision=0.12.0
# Add conda-lock (and other packages, as needed) to pyproject.toml and poetry.lock
poetry add --lock conda-lock

# Remove the bootstrap env
conda deactivate
rm -rf /tmp/bootstrap

# Add Conda spec and lock files
git add environment.yml virtual-packages.yml conda-linux-64.lock
# Add Poetry spec and lock files
git add pyproject.toml poetry.lock
git commit

Usage

The above setup may seem complex, but it can be used in a fairly simple way.
Creating the environment

conda create --name my_project_env --file conda-linux-64.lock
conda activate my_project_env
poetry install

Activating the environment

conda activate my_project_env

Updating the environment

# Re-generate Conda lock file(s) based on environment.yml
conda-lock -k explicit --conda mamba
# Update Conda packages based on re-generated lock file
mamba update --file conda-linux-64.lock
# Update Poetry packages and re-generate poetry.lock
poetry update

If building a new environment make sure to install pomegranate with pip and then copy the 
requirements exactly into the pyproject.toml file.


.. autosummary::
   :toctree: generated
   :recursive:

   pybkb
