from setuptools import setup, find_packages

setup(
    name='sweet-rl',
    version='0.0.1',
    url='https://github.com/Hadjubuntu/sweet-rl',
    author='Adrien Hadj-Salah',
    author_email='adrien.hadj.salah@gmail.com',
    description='The sweetest Reinforcement Learning framework',
    packages=find_packages(),  
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.18.1',
        'tensorflow==2.1.0',
        'pytorch==1.0.2',
        'gym==0.15.4',
        'pandas==0.25.3',
        'pytest==5.2.2',
        'matplotlib==3.1.2'],
)