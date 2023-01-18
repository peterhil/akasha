# Installation on FreeBSD

### System library requirements

Install the required C libraries:

    sudo pkg install portmidi sdl2 libsndfile libsamplerate

### Python packages

Scipy has not been working on FreeBSD since a long time because wheel
building fails. So create virtuenvs with the `--system-site-packages`
and install the `py39-scipy` package. Scikit-image also fails to build.

    sudo pkg install py39-scipy
    sudo pkg install py39-scikit-image
    python -m venv --system-site-packages venv/py39-system
