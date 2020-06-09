
from setuptools import find_packages, setup


with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()

setup(

    name = 'kprototypes',
    version = '0.1.2',
    packages = find_packages(),

    author = 'Johan Berdat',
    author_email = 'jojolebarjos@gmail.com',
    license = 'MIT',

    url = 'https://gitlab.com/jojolebarjos/kprototypes',

    description = 'k-prototypes for numerical and categorical clustering',
    long_description = long_description,
    long_description_content_type = 'text/markdown',

    keywords = [
        'clustering',
        'k-prototypes',
        'k-means',
        'k-modes',
    ],

    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    install_requires = [
        #'fastkde', # TODO wait until broken setup is solved
        'numba',
        'numpy',
        'scikit-learn',
    ],

)
