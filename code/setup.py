from setuptools import setup, find_packages


setup(
    name="M2-IABD-DeepRL",
    version="1.0",
    packages=find_packages(),
    install_requires=['gym>=0.21.0','numpy','matplotlib','pillow','pyqt5', 'torch==1.10', 'einops']
)
