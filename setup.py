from setuptools import setup, find_packages

setup(
    name='nn-monitor',
    version='1.2.0',
    description='Neural Network Training Monitoring Framework',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'torch>=1.9',
        'matplotlib>=3.3',
    ],
    extras_require={
        'lgbm': ['lightgbm>=3.0', 'scikit-learn>=1.0'],
        'hmm': ['hmmlearn>=0.3', 'scipy>=1.7'],
        'trackers': ['tensorboard>=2.8'],
        'all': ['lightgbm>=3.0', 'scikit-learn>=1.0', 'hmmlearn>=0.3',
                'scipy>=1.7', 'tensorboard>=2.8'],
    },
)
