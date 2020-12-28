# Akasha installation on Ubuntu Linux

## 1. Create a virtualenv and activate it

```
virtualenv -p python2.7 --system-site-packages venv/py27
. ./venv/py27/bin/activate
```

## 2. Install the Pygame dependent SDL and other libraries

```
sudo apt-get install libsdl1.2-dev \
  libsdl-image1.2-dev \
  libsdl-mixer1.2-dev \
  libsdl-ttf2.0-dev
sudo apt-get install libportmidi-dev \
  libsndfile1-dev \
  portaudio19-dev \
  libsamplerate0-dev \
  libsmpeg-dev
```

## 3. Install the Python libraries using Pip

```
cd akasha-git
pip install -r install/dev-requires.pip
```
