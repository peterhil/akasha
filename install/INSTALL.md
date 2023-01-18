# Akasha installation

Akasha depends on a number of software libraries.
Follow these steps to install the dependencies.

## 1. Change directory into the Akasha Git repository root

    cd Akasha/akasha-git


## 2. Install Python and create a virtualenv

### Notes

1. **Important:** Install Python 3 versions on a virtualenv using [venv] module,
   because [Matplotlib needs] a framework build of Python. With Python 2, the
   only option seems to be using another Matplotlib backend on Mac OS X, unless
   a [Framework Python] is installed separately and used.

2. Python versions before 3.5.3 only [support openssl@1.0]. To install
   Python 3.4 with pyenv and homebrew:

        PYTHON_BUILD_HOMEBREW_OPENSSL_FORMULA=openssl@1.0 \
        pyenv install -v 3.4.10

3. Current versions of `scikits.audiolab` and/or `scikits.resample` do not seem to work well with Python 3.6.

### Using Pyenv

    pyenv install 2.7.18
    pyenv virtualenv 2.7.18 akasha-27
    workon akasha-27

    pyenv install 3.6.13
    pyenv virtualenv -f --python python3.6 --pip 21.1.2 3.6.13 akasha-36
    workon akasha-36

[Building Framework Python] on MacOS:

    env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.6.13

Not recommended for Python 2, see notes below.

### Using venv module

    python3.6 -m venv --prompt akasha-36 venv/py36
    . ./venv/py36/bin/activate

 Recommended for Python 3

### Using virtualenv with Python 2

    virtualenv -p python2.7 --prompt '(akasha-27) ' venv/py27
    . ./venv/py27/bin/activate

Use graphically installed [Framework Python] or some other backend than `macosx` or `wxagg` on Matplotlib!

### Using miniconda

    # Install miniconda, and then:
    conda update -n base -c defaults conda
    conda create -n akasha-27 python=2.7 pip
    conda init zsh
    conda activate akasha-27
    pip install -r install/dev-requires.pip
    conda install python.app
    # Then somehow use ipython with pythonw?!

[Building Framework Python]: https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with-framework-support-on-os-x
[Framework Python]: https://docs.python.org/3/using/mac.html
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
    brew install portaudio
    brew install gcc  # Latest GCC versions include gfortran


## 5. Install the Python libraries using Pip

    pip install --use-pep517 -r install/requires.pip  # Or
	pip install -r install/dev-requires.pip  # For development
	pip install -r install/extra-requires.pip  # For development extras

## 6. Start experimenting with bpython or ipython

```
from akasha.lab import *

e = Exponential(-0.957, amp=0.75)
s = Super(5, 1.7, 1.7, 1)
o = Osc(108, curve=s)
h = Overtones(o, n=5, rand_phase=False)
snd = Mix(e, h)

# Animate and play
anim(snd, antialias=True, lines=True)

# Graph signal
graph(snd, antialias=True, lines=True)

# Plot signal using Matplotlib
plot_signal(snd[:5*sampler.rate])
```
