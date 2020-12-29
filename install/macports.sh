#!/bin/bash

# Pillow depends
port install freetype
# Pygame depends
port install portmidi libsdl-framework libsdl_mixer-framework libsdl_ttf-framework libsdl_image-framework
# python-soundfile depends
port install libsndfile
# scikits.samplerate depends
port install libsamplerate
