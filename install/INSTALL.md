# Akasha installation

Akasha depends on a number of software libraries. To install all the required software do:

## 1. Change directory into the Akasha Git repository root

    cd Akasha/Resonance-git

## 2. Install XQuartz

Download and install the [XQuartz](http://xquartz.macosforge.org/landing/) X11 implementation
for Pygame and Tk windowed Python apps.

## 3. Make sure your Python version is using a framework build

Matplotlib will not work in a virtual environment unless using a
[framework build of Python](https://matplotlib.org/faq/osx_framework.html).

## 4. Create a virtualenv and activate it

    virtualenv -p python2.7 --system-site-packages venv/py27
    . ./venv/py27/bin/activate

Install Python 3 versions on a virtualenv using venv module, because [Matplotlib =>1.5 does not support virtualenv easily](https://matplotlib.org/faq/osx_framework.html):

    python3.4 -m venv venv/py34
    . ./venv/py34/bin/activate

Python3 [versions before 3.5.3 only support openssl@1.0](https://github.com/pyenv/pyenv/issues/950). So to install Python 3.4 with pyenv and brew (scikits.audiolab and/or scikits.resample do not seem to work with Python 3.6):

    PYTHON_BUILD_HOMEBREW_OPENSSL_FORMULA=openssl@1.0 \
    pyenv install -v 3.4.10

## 5. Install the Pygame dependent SDL libraries with Homebrew or MacPorts

    # Macports

    # sudo port install libsdl-framework libsdl_mixer-framework libsdl_image-framework libsdl_ttf-framework
    sudo port install portmidi libsdl libsdl_mixer libsdl_ttf libsdl_image
    hg clone https://bitbucket.org/pygame/pygame
    mv pygame pygame-hg
    cd pygame-hg
    # export LDFLAGS="-L/opt/local/Library/Frameworks -L/opt/local/lib"
    # export CFLAGS="-I/opt/local/include/SDL -I/opt/local/include"
    sudo python setup.py install  # This method worked on 2014-09-05 without VirtualEnv!!!

    # Homebrew

    # Note! SDL Needs to be installed with Homebrew, the Macports version fails to put SDL.h on path
    brew install sdl sdl_image sdl_mixer sdl_ttf
    brew install portmidi  # Optional dependency for Pygame
    brew install libogg libpng

## 6. Install the other required libraries with Macports

    # Pillow depends (install with Macports)
    port install freetype

    # Scikits-audiofile and scikits-samplerate depends
    brew install libsndfile libsamplerate
    brew install portaudio
    brew install gcc  # Latest GCC versions include gfortran

## 7. Install the Python libraries using Pip

	pip install -r install/requires.pip  # Or
	pip install -r install/dev-requires.pip  # For development
	pip install -r install/extra-requires.pip  # For development extras
