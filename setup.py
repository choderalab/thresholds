from setuptools import setup, find_packages

setup(
    name='thresholds',
    version='0.1',
    description='Find maximum tolerable timesteps for molecular simulation.',
    author='Josh Fass, John Chodera',
    author_email='{josh.fass, john.chodera}@choderalab.org',
    packages=find_packages(),
    keywords=['molecular simulation', 'probabilistic bisection'],
    url='https://github.com/choderalab/thresholds',
    license='MIT',
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3'],
)
