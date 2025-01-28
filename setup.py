from setuptools import find_packages, setup

setup(
    name="pix2pix",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.12.0",
        "gcsfs",
        "google-cloud-storage",
        "matplotlib",
        "IPython",
    ],
)
