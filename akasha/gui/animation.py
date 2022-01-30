#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Animation module
"""

from __future__ import division

import numpy as np

from akasha.audio.frequency import Frequency, octaves
from akasha.audio.mixins.releasable import Releasable
from akasha.graphic.drawing import get_canvas
from akasha.gui.widgets import ComplexView, VideoTransferView
from akasha.gui.pygame_adapter import PygameGui
from akasha.math import div_safe_zero, pcm, minfloat
from akasha.settings import config
from akasha.timing import sampler, Timed, Watch
from akasha.tunings import PianoLayout, WickiLayout
from akasha.utils.log import logger

keyboard = WickiLayout()
# keyboard = PianoLayout()


def anim(
    snd,
    size=800,
    name='Resonance',
    gui=PygameGui,
    antialias=False,
    lines=True,
    colours=True,
    mixer_options=(),
    style='complex',
):
    """
    Animate complex sound signal
    """
    logger.info(
        "Akasha animation is using %s Hz sampler rate and "
        "%s fps video rate.",
        sampler.rate,
        sampler.videorate,
    )
    sampler.paused = True if hasattr(snd, 'frequency') else False
    gui = gui()
    screen = gui.init(name, size)
    channel = gui.init_mixer(*mixer_options)

    if style == 'complex':
        widget = ComplexView(
            screen,
            size,
            antialias=antialias,
            lines=lines,
            colours=colours,
        )
    elif style == 'transfer':
        widget = VideoTransferView(
            screen, size=size, standard='PAL', axis='real'
        )
    else:
        logger.error("Unknown animation style: '%s'", style)
        gui.cleanup()
        return False

    return loop(gui, snd, channel, widget)


def loop(gui, snd, channel, widget):
    t = gui.tick(sampler.videorate)
    watch = Watch()
    last = 0

    # Handle first time, draw axis and bg
    first_time = True
    widget.render(np.array([0, 0, 0], dtype=np.complex128))
    gui.flip()

    while True:
        try:
            with Timed() as loop_time:
                input_time, audio_time, video_time = 0, 0, 0

                with Timed() as input_time:
                    for event in gui.get_event():
                        if handle_input(gui, snd, watch, event):
                            if (
                                first_time
                                and hasattr(snd, 'frequency')
                                and sampler.paused
                            ):
                                sampler.pause()
                            watch.reset()
                            last = 0
                if not sampler.paused:
                    current = watch.next()
                    current_slice = slice(
                        sampler.at(last, np.int), sampler.at(current, np.int)
                    )
                    samples = snd[current_slice]

                    if (
                        len(samples) == 0
                        and current_slice.start != current_slice.stop
                    ):
                        raise SystemExit('Sound ended!')
                    if len(samples) > 0:
                        with Timed() as audio_time:
                            gui.queue_audio(samples, channel)
                        with Timed() as video_time:
                            widget.render(samples)
                        gui.flip()
                        last = current
            if not sampler.paused:
                percent = float(loop_time) / (1.0 / sampler.videorate) * 100
                av_time = float(audio_time) + float(video_time)
                av_percent = av_time / float(loop_time) * 100
                fps = watch.get_fps()
                if percent >= config.logging_limits.LOOP_THRESHOLD_PERCENT:
                    logger.warning(
                        "Anim: (%d) %.1f%% %.1f fps,\tloop: %.1fhz "
                        "(%.1f%%) video: %.1fhz audio: %.1fkhz io: %.1fkhz",
                        t,
                        percent,
                        fps,
                        div_safe_zero(1, loop_time),
                        av_percent,
                        div_safe_zero(1, video_time),
                        div_safe_zero(1e-3, audio_time),
                        div_safe_zero(1e-3, input_time),
                    )
            t = gui.tick(sampler.videorate)
        except KeyboardInterrupt:
            logger.info("Got KeyboardInterrupt (CTRL-C)!")
            break
        except SystemExit as err:
            logger.info("Ending animation: %s", err)
            break
    gui.cleanup()
    if isinstance(snd, Releasable):
        snd.release_at(None)
    return watch


def handle_input(gui, snd, watch, event):
    """
    Handle pygame events.
    """
    # Quit
    if gui.key_escape(event):
        raise SystemExit('Quit.')

    # Pause
    if gui.key_pause(event):
        sampler.pause()
        watch.pause()
        return

    # Key down
    elif gui.keydown(event):
        # logger.debug("Key '%s' (%s) down.", gui.keyname(event), event.key)
        step_size = 5 if gui.key_shift(event) else 1
        # Rewind
        if gui.key_f7(event):
            if isinstance(snd, Releasable):
                snd.release_at(None)
            logger.info("Rewind")
            return True  # reset
        # Arrows
        elif gui.key_up(event):
            if gui.key_alt(event):
                sampler.change_frametime(rel=step_size)
            else:
                # keyboard.move(-2, 0)
                keyboard.base = np.clip(
                    keyboard.base * 2.0,
                    a_min=Frequency(octaves()[2]),
                    a_max=Frequency(octaves()[-2]),
                )
        elif gui.key_down(event):
            if gui.key_alt(event):
                sampler.change_frametime(rel=-step_size)
            else:
                # keyboard.move(2, 0)
                keyboard.base = np.clip(
                    keyboard.base / 2.0,
                    a_min=Frequency(octaves()[2]),
                    a_max=Frequency(octaves()[-2]),
                )
        elif gui.key_left(event):
            keyboard.move(0, 1)
        elif gui.key_right(event):
            keyboard.move(0, -1)
        # Change frequency
        elif hasattr(snd, 'frequency'):
            change_frequency(snd, event.key)
            return True  # reset
        logger.debug('Keyboard base: %s', keyboard.base)
    # Key up
    elif gui.keyup(event) and hasattr(snd, 'frequency'):
        if gui.key_caps_lock(event):
            change_frequency(snd, event.key)
            return True  # reset
        else:
            if isinstance(snd, Releasable):
                try:
                    snd.release_at(watch.time())
                except TypeError:
                    logger.warning(
                        "Can't get current value from the iterator."
                    )
                    snd.release_at(None)
                # logger.debug(
                #     "Key '%s' (%s) up, released at: %s",
                #     gui.keyname(event), event.key, snd.released_at
                # )
        return
    # Mouse
    elif hasattr(snd, 'frequency') and gui.mouse_event(event):
        if gui.mouse_down(event):
            snd.pitch_bend = 0
        elif gui.mouse_up(event):
            snd.pitch_bend = None
        elif (
            gui.mouse_motion(event)
            and getattr(snd, 'pitch_bend', None) is not None
        ):
            size = gui.get_size()[1] or 0
            snd.pitch_bend = event.pos[1]

            logger.debug("Pitch bend == event.pos[0]: %s", snd.pitch_bend)

            odd = (size + 1) % 2
            norm_size = (size // 2) * 2 + odd
            scale = np.logspace(
                -1 / 4, 1 / 4, norm_size, endpoint=True, base=2
            )
            ratio = scale[snd.pitch_bend]

            new_freq = np.clip(
                snd.frequency * ratio,
                a_min=-minfloat()[0],
                a_max=sampler.nyquist,
            )
            snd.frequency = new_freq
            logger.info(
                "Pitched frequency to %s (with ratio %.04f) position %s, "
                "bend %s. [%s, %s, %s]",
                snd.frequency,
                ratio,
                event.pos[1],
                snd.pitch_bend,
                scale[0],
                scale[len(scale) // 2],
                scale[-1],
            )
    # else:
    #     logger.debug("Other: %s", event)
    return


def change_frequency(snd, key):
    """
    Change frequency of the sound based on key position.
    """
    frequency = np.clip(
        keyboard.get_frequency(key),
        a_min=Frequency(octaves()[0]),
        a_max=Frequency.from_ratio(1, 2),
    )
    snd.frequency = frequency or snd.frequency
    if isinstance(snd, Releasable):
        snd.release_at(None)
    logger.debug(
        "Changed frequency: %s.",
        snd.frequency if hasattr(snd, 'frequency') else snd,
    )

    return snd.frequency
