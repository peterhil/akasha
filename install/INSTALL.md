# Akasha installation

Akasha depends on a number of software libraries.
Follow these steps to install the dependencies.

## 1. Change directory into the Akasha Git repository root

    cd Akasha/akasha-git


## 2. Install Python and create a virtualenv

Using Pyenv (recommended):

    pyenv install 2.7.18
    pyenv virtualenv -p python2.7 2.7.18 akasha-27
    workon akasha-27

    pyenv install 3.6.10
    pyenv virtualenv --creator venv -p python3.6 3.6.10 akasha-36
    workon akasha-36

Using virtualenv or venv module (old way):

    virtualenv -p python2.7 --system-site-packages venv/py27
    . ./venv/py27/bin/activate

    python3.4 -m venv venv/py34
    . ./venv/py34/bin/activate

**Notes:**

1. Install Python 3 versions on a virtualenv using [venv] module, because
   [Matplotlib needs] a framework build of Python.

2. Python versions before 3.5.3 only [support openssl@1.0]. To install
   Python 3.4 with pyenv and homebrew:

    PYTHON_BUILD_HOMEBREW_OPENSSL_FORMULA=openssl@1.0 \
    pyenv install -v 3.4.10

3. Current versions of scikits.audiolab and/or scikits.resample do not seem to
   work well with Python 3.6.

[Matplotlib needs]: https://matplotlib.org/faq/osx_framework.html
[support openssl@1.0]: https://github.com/pyenv/pyenv/issues/950
[venv]: https://docs.python.org/3/library/venv.html


## 3. Install the Pygame dependent SDL libraries

### Using Homebrew

    # Note! SDL Needs to be installed with Homebrew, the Macports version fails to put SDL.h on path
    brew install sdl sdl_image sdl_mixer sdl_ttf
    brew install portmidi  # Optional dependency for Pygame
    brew install libogg libpng

### Using Macports

    # sudo port install libsdl-framework libsdl_mixer-framework libsdl_image-framework libsdl_ttf-framework
    sudo port install portmidi libsdl libsdl_mixer libsdl_ttf libsdl_image

    hg clone https://bitbucket.org/pygame/pygame
    cd pygame
    # export LDFLAGS="-L/opt/local/Library/Frameworks -L/opt/local/lib"
    # export CFLAGS="-I/opt/local/include/SDL -I/opt/local/include"
    sudo python setup.py install  # This method worked on 2014-09-05 without VirtualEnv!!!


## 4. Install other required libraries

    # Pillow depends (install with Macports)
    port install freetype

    # Scikits-audiofile and scikits-samplerate depends
    brew install libsndfile libsamplerate
    brew install gcc  # Latest GCC versions include gfortran


## 5. Install the Python libraries using Pip

	pip install -r install/requires.pip  # Or
	pip install -r install/dev-requires.pip  # For development
	pip install -r install/extra-requires.pip  # For development extras

## 6. Start experimenting

    ipython

    from akasha.lab import *
    e = Exponential(-0.987, amp=0.9)
    s = Super(6, 1.5, 1.5, 1.5)
    o = Osc(220, curve=s)
    h = Harmonics(o, n=1, rand_phase=False)
    snd = Mix(e, h)

    # Animate and play
    anim(snd, antialias=True, lines=True)

    # Graph signal
    graph(snd, antialias=True, lines=True)

    # Plot signal using Matplotlib
    plot_signal(snd[:5*sampler.rate])
