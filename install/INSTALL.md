# Akasha installation

Akasha depends on a number of software libraries. To install all the required software do:

## 1. Change directory into the Akasha Git repository root

    cd Akasha/Resonance-git

## 2. Install XQuartz

Download and install the [XQuartz](http://xquartz.macosforge.org/landing/) X11 implementation
for Pygame and Tk windowed Python apps.

## 3. Create a virtualenv and activate it

    virtualenv -p python2.7 --system-site-packages venv/py27
    . ./venv/py27/bin/activate

## 4. Install the Pygame dependent SDL libraries with Homebrew or MacPorts

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

## 5. Install the other required libraries with Macports

    # Pillow depends (install with Macports)
    port install freetype

    # Scikits-audiofile and scikits-samplerate depends
    brew install libsndfile libsamplerate
    brew install gcc  # Latest GCC versions include gfortran

## 6. Install the Python libraries using Pip

	pip install -r install/requires.pip
	pip install -r install/test-requires.pip  # Optional for development
