from setuptools import setup, find_packages

setup(
    name='sweet-rl',
    version='0.0.1',
    url='https://github.com/Hadjubuntu/sweet-rl',
    author='Adrien Hadj-Salah',
    author_email='adrien.hadj.salah@gmail.com',
    description='A sweet and nice reinforcement learning framework',
    packages=find_packages(),    
    python_requires='>=3.8',
    install_requires=[
        'numpy==1.17.3', 
        'tensorflow==2.0.0',
        'gym=0.15.4',
        'keras==2.3.1',
        'pandas==0.25.3',
        'pytest==5.2.2'],
)