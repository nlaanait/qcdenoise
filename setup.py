from setuptools import setup, find_packages

setup(
    name='qcdenoise',
    version='0.1',
    packages=['qcdenoise'],
    url='',
    license='',
    author='Numan Laanait',
    author_email='nlaanait@gmail.com',
    description='',
    install_requires=[
        'numpy',
        'qiskit',
        'torch',
        'lmdb',
        'pytest',
        'pytorch-lightning'],
    # test_suite='tests',
    python_requires='>=3.6',
    package_dir={'qcdenoise': 'qcdenoise'},
    package_data={'qcdenoise': ['data']},
    include_package_data=True
)
