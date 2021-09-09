from setuptools import setup, find_packages

setup(
    name='raft',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'matplotlib',
        'tensorboard',
        'scipy'
    ]
)

