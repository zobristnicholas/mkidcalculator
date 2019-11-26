# mkidcalculator
This package provides tools for analyzing MKID data loaded from arbitrary sources. Routines are provided along with plotting tools to make basic analysis tasks simple and straightforward. 

## Getting Started
These instructions will help you install _mkidcalculator_ on your machine. The code is still in it's beta stage so there is currently no release version. It must be downloaded and installed directly from GitHub.

### Prerequisites
The code is designed to run on all python versions greater than 3.7. All other python prerequisites will be automatically downloaded during the installation. 

Git must be installed to clone the repository (as in the install instructions), but it can be downloaded directly from [GitHub](https://github.com/zobristnicholas/mkidcalculator) as well.

No testing is currently implemented to verify cross platform compatibility, but the code is expected to be platform independent. Development was done on Mac OSX.  

### Installing
On the command line run the following with your choice of \<directory\> and \<version\>:
```
cd <directory>
git clone --branch <version> https://github.com/zobristnicholas/mkidcalculator.git
pip install -e mkidcalculator
```
- In the first line choose the directory where you want the code to exist.
- In the second line choose the [version](https://github.com/zobristnicholas/mkidcalculator/tags) that you want to install. (e.g. 0.3)
- The third line will install the code. Checking out other versions with git (```git checkout <version>```) will automatically update the Python installation.

## Versions
The versions of this code follow the [PEP440](https://www.python.org/dev/peps/pep-0440/) specifications.

Each version is given a git [tag](https://github.com/zobristnicholas/mkidcalculator/tags) corresponding to a particular commit. Only these commits should be used since the others may not be stable. Versions prior to 1.0 should be considered in the beta stage and subject to changes in the code's API.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Much of the loop fitting was developed from the [scraps package](https://github.com/FaustinCarter/scraps), but many other areas of this project have benefited from the work of other people. Care has been taken to specify where these contributions are in the documentation for the functions that use them.
