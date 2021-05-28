from setuptools import setup, find_packages

setup(
    name='qcdenoise',
    version='0.1',
    packages=['qcdenoise'],
    url='https://github.com/nlaanait/qcdenoise',
    license='',
    author=['Numan Laanait', 'Kate Hamilton'],
    author_email=['nlaanait@gmail.com'],
    description='',
    install_requires=[
        'numpy',
        'networkx',
        'qiskit',
        'torch',
        'lmdb',
        'pytest',
        'pytorch-lightning',
        'uncertainties'],
    test_suite='tests',
    python_requires='>=3.6',
    package_dir={'qcdenoise': 'qcdenoise'},
    package_data={'qcdenoise': ['data']},
    include_package_data=True
)
