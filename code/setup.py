from setuptools import setup, find_packages


setup(
    name="M2-IABD-DeepRL",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'gym>=0.21.0',
        # 'gym[atari]', 'gym[accept-rom-license]',
        'opencv-python','numpy','matplotlib','pillow','pyqt5==5.15', 'torch>=2.6.0', 'einops', 'tensorboard',
        'gymnasium', 'shimmy[gym-v21]', 'shimmy',
        'gymnasium[other]',
        'typing_extensions==4.11.0',
        'torchvision==0.21.0',
        'moviepy==1.0.3',
    ]
)
