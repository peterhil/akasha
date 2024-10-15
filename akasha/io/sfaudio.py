import numpy as np
import pyaudio
import soundfile as sf

from scipy.signal import hilbert

from akasha.audio.channels import num_channels
from akasha.funct.itertools import blockwise
from akasha.io.path import file_extension, relative_path
from akasha.timing import sampler, time_slice


def play(
    sndobj, dur=5.0, start=0, axis='real', fs=sampler.rate, buffer_size=512
):
    """
    Play a sound using PyAudio bindings for portaudio.
    """
    pa = pyaudio.PyAudio()
    time = time_slice(dur, start)
    if isinstance(sndobj[0], np.floating):
        axis = 'real'
    data = getattr(sndobj[time], axis).astype(np.float32)

    n_channels = num_channels(data)
    if n_channels > 2:
        # TODO Use mapping argument with python-sounddevice library
        # to overcome this or DIY?
        raise NotImplementedError(
            'Only mono and stereo sounds are supported for now'
        )

    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=num_channels(data),
        rate=fs,
        frames_per_buffer=buffer_size,
        output=True,
    )

    for frame in blockwise(data, buffer_size):
        stream.write(frame.flatten().tobytes(), num_frames=frame.shape[0])

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

    if filename[0] != '/':  # Relative path
        filename = '/'.join([sdir, filename])

    extension = file_extension(filename)
    check_format(extension)

    # time = time_slice(dur, start)

    # Read data
    data, samplerate = sf.read(
        filename,
        frames=int(dur * sampler.rate),
        start=int(start * sampler.rate),
        dtype=np.float64,
    )

    if complex:
        return hilbert(data, axis=0)  # TODO Move to anim
    else:
        return data


def write(
    sndobj,
    filename='test_sound',
    dur=5.0,
    start=0,
    axis='real',
    fs=sampler.rate,
    fmt='WAV',
    subtype=None,
    endian=None,
    sdir=relative_path('../../Sounds/Out/'),
):
    """
    Write a sound file.

    Most common formats:
    - WAV (or W64, RF64)
    - AIFF
    - AU

    Subtype can be:
    - PCM_S8, PCM_16, PCM_24, PCM_32 (or PCM_U8 for WAV and RAW only)
    - FLOAT, DOUBLE
    - ULAW, ALAW, and many others

    See python-soundfile for all available format, subtype and endian options:
    https://github.com/bastibe/python-soundfile/blob/master/soundfile.py#L38
    """
    if filename[0] != '/':
        filename = '/'.join([sdir, filename])  # Relative path

    extension = file_extension(filename)
    if not extension:
        if axis == 'imag':
            filename = filename + '_' + axis
        filename = filename + '.' + fmt.lower()

    time = time_slice(dur, start)
    data = getattr(sndobj[time], axis)

    sf.write(
        filename,
        data,
        fs,
        format=fmt,
        subtype=subtype,
        endian=endian,
    )


def check_format(format):
    """Checks that a requested format is available (in libsndfile)."""
    available = sf.available_formats().keys()
    if format.upper() not in available:
        raise ValueError(
            f"File format '{format!s}' not available. Try one of: {available}"
        )
