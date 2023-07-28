from setuptools import setup, find_packages
import toml

# Read the content of requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the content of pyproject.toml file
with open('pyproject.toml') as f:
    pyproject = toml.load(f)

# Extract information from pyproject.toml
package_info = pyproject['tool']['poetry']
name = package_info['name']
version = package_info['version']
author = package_info['authors'][0]
description = package_info['description']

setup(
    name=name,
    version=version,
    author=author,
    description=description,
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=[
        'https://bitbucket.org/yakaboskic/pygobnilp.git@38c178e52944fe427c0aad6721569d392494f9d3#egg=pygobnilp',
        ]
)
