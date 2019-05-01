from setuptools import setup, find_packages

version_number = '0.0'


setup(name='mkidcalculator',
      description='Tools for analyzing MKID data',
      version=version_number,
      author='Nicholas Zobrist',
      license='GPLv3',
      url='http://github.com/zobristnicholas/mkidcalculator',
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'lmfit',
                        'matplotlib',
                        'corner'],
      classifiers=['Development Status :: 1 - Planning',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Physics'])
