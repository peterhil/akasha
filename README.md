# Akasha

My exploration into audio programming and DSP algorithms inspired by the
writings of [Julius Orson Smith III](https://ccrma.stanford.edu/~jos/) of
Stanford’s [CCRMA](https://ccrma.stanford.edu/) (Center for Computer Research in Music and Acoustics). I also use complex numbers for almost all the calculations.

I really started to learn programming in 2005, and this project inspired me to
continue learning. I first started this project using Ruby and [NArray](https://masa16.github.io/narray/), but in 2009 I started to use Python at work, so I ported it to Python and Numpy.

Of course I realised at some point that Python would never be suitable for
realtime audio because of the GIL, and learned [D](https://dlang.org/), [Common Lisp](https://www.lispworks.com/documentation/HyperSpec/Front/Contents.htm) and
Haskell. Some time later I also learned some Rust, Zig and Nim. Maybe some day I learn C++ properly.

Please be warned to only use Akasha for prototyping audio DSP
algorithms and trying things out, and not expect to make anything playble
using Python.

The functions in the module `akasha.math.functions` might be generally useful.


## Installation

Akasha should work on 
[macOS](./install/INSTALL.md), 
[Linux](./install/INSTALL-linux.md) and
[FreeBSD](./install/Install-on-FreeBSD.md).

See the `/install` directory for various installation instructions and pip
requirement files.

On macOS, it is recommended to use the framework versions of the SDL
libraries.

In recent years, I have only used Macports to install the required
dependencies instead of Homebrew, because I find the latters engineering
quality to be dubious.


## Usage

Basic usage with IPython:

    source ./venv/py310/bin/activate
    ipython
    from akasha.lab import *

Then explore the `akasha` module and especially the `audio`, `dsp` and `io`
submodules.

### Reading audio files

    In [1]: from akasha.lab import *
    In [2]: bj = read('Bjork - Human Behaviour.wav', dur=120, sdir='../../Sounds/_Music samples/')
    In [3]: snd = bj[..., 0]  # Make the sound mono
    In [4]: play(snd)
    In [5]: anim(snd)


## Develop

Run tests:

    pytest -k akasha akasha


## License

This work is licensed under the [MPL 2.0 License](./LICENSE.txt).

&copy; 2005–2025 Peter Hillerström
