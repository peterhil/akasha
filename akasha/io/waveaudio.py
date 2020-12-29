import numpy as np
import pyaudio
import sys

from scipy.signal import hilbert
from wavefile import Format, WaveReader, WaveWriter

from akasha.funct import blockwise
from akasha.io import file_extension, relative_path
from akasha.timing import sampler, time_slice
from akasha.utils.log import logger


pa = pyaudio.PyAudio()
default_format = Format.AIFF | Format.PCM_16 | Format.ENDIAN_FILE


def play(
        sndobj,
        dur=5.0,
        start=0,
        axis='imag',
        fs=sampler.rate,
        buffer_size=512
    ):
    """
    Play a sound using PyAudio bindings for portaudio.
    """
    time = time_slice(dur, start)
    if isinstance(sndobj[0], np.floating):
        axis = 'real'
    data = getattr(sndobj[time], axis).astype(np.float32)

    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=fs,
        frames_per_buffer=buffer_size,
        output=True,
    )

    for frame in blockwise(data, buffer_size):
        stream.write(frame.flatten(), frame.shape[-1])

    stream.close()


def read(
        filename,
        dur=5.0,
        start=0,
        fs=sampler.rate,
        complex=True,
        sdir=relative_path('../../Sounds/_Music samples/'),
    ):
    """
    Read a sound file.
    """
    # TODO Do some conversion if sampling rates differ?

    if filename[0] != '/':
        filename = '/'.join([sdir, filename])  #Relative path

    extension = file_extension(filename)
    # check_format(extension)

    time = time_slice(dur, start)

    r = WaveReader(filename)
    wave_info(r)

    r.seek(int(start * sampler.rate))

    data = np.zeros((r.channels, dur * sampler.rate), np.float32, order='F')
    r.read(data)
    r.close()

    # Make mono
    # TODO Enable stereo and multichannel
    if data.ndim > 1:
        data = data[-1]

    if complex:
        return hilbert(data)  # TODO Move to anim
    else:
        return data


def write(
        sndobj,
        filename='test_sound',
        fmt=default_format,
        dur=5.0,
        start=0,
        axis='imag',
        fs=sampler.rate,
        sdir=relative_path('../../Sounds/Out/'),
    ):
    """
    Write a sound file.
    """
    if filename[0] != '/':
        filename = '/'.join([sdir, filename])  # Relative path

    # filename = filename + '_' + axis + '.' + fmt.file_format

    time = time_slice(dur, start)
    data = np.atleast_2d(getattr(sndobj[time], axis))

    print('Data shape:', data.shape)

    if np.ndim(data) <= 1:
        n_channels = 1
    elif np.ndim(data) == 2:
        n_channels = data.shape[1]
    else:
        RuntimeError("Only rank 0, 1, and 2 arrays supported as audio data")

    with WaveWriter(
        filename,
        channels=n_channels,
        format=fmt,
    ) as w:
        w.write(data)


def wave_info(wav):
    logger.debug("Reading PCM audio:")
    logger.debug("Title: {}".format(wav.metadata.title))
    logger.debug("Artist: {}".format(wav.metadata.artist))
    logger.debug("Channels: {}".format(wav.channels))
    logger.debug("Format: 0x%x" % wav.format)
    logger.debug("Sample Rate: {}".format(wav.samplerate))
    logger.debug("Frames: {}".format(wav.frames))
