from setuptools import setup, find_packages


setup(
    name="M2-IABD-DeepRL",
    version="1.0",
    packages=find_packages(),
    install_requires=['gym','numpy','matplotlib','pillow','pyqt5', 'torch']
)
