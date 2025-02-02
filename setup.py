from setuptools import find_packages, setup

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0",
        "numpy",
        "gcsfs",
        "google-cloud-storage",
        "matplotlib",
        "IPython",
    ],
)
