from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sweet-rl',
    version='0.1',
    url='https://github.com/Hadjubuntu/sweet-rl',
    author='Adrien Hadj-Salah',
    author_email='adrien.hadj.salah@gmail.com',
    description='The sweetest Reinforcement Learning framework',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.18.1',
        'tensorflow==2.11.1',
        'torch==1.4.0',
        'gym==0.15.4',
        'pandas==0.25.3',
        'pytest==5.2.2',
        'matplotlib==3.1.2',
        'pytest-cov==2.8.1',
        'jupyter==1.0.0'],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
