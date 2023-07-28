# Use the official Python image as the base image
FROM python:3.11

# Install graphviz
RUN apt update -y
RUN apt install graphviz graphviz-dev -y

# Update pip and install requirements
RUN pip install --upgrade pip
RUN pip install toml setuptools


# Set the working directory in the container
WORKDIR /pybkb

# Copy requirements and setup over into the container at /pybkb
# Copy full project over
COPY . /pybkb

# Install requirements using pip
RUN pip install -r requirements.txt


# Install your project
RUN python setup.py install


# Define the command to run your application (if applicable)
# For example, if your application has a script named "main.py":
# CMD ["python", "main.py"]
