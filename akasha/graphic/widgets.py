from akasha.graphic.drawing import get_canvas, blit, draw, video_transfer


class ComplexView(object):
    """
    Show a sound signal on screen.
    """
    def __init__(self, screen, antialias=True, lines=False, colours=True):
        self._surface = screen
        self.antialias = antialias
        self.lines = lines
        self.colours = colours

    def render(self, signal):
        img = draw(
            signal,
            self._surface.get_size()[0],
            antialias=self.antialias,
            lines=self.lines,
            colours=self.colours,
            axis=True,
            screen=self._surface
        )

        if img is not None:
            blit(self._surface, img)


class VideoTransferView(object):
    """
    Show a sound signal using the old video tape audio recording technique.
    See: http://en.wikipedia.org/wiki/44100_Hz#Recording_on_video_equipment
    """
    def __init__(self, screen, size=720, standard='PAL', axis='real'):
        self._surface = screen
        self.size = size
        self.standard = standard
        self.axis = axis

    def render(self, signal):
        size = self._surface.get_size()[0]
        img = get_canvas(size)
        tfer = video_transfer(
            signal,
            standard=self.standard,
            axis=self.axis,
            horiz=self.size
        )

        black = int(round((size - tfer.shape[0]) / 2.0))
        img[:, black:-black, :] = tfer[:, :img.shape[1], :].transpose(1, 0, 2)

        blit(self._surface, img)
