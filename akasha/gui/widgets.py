from akasha.graphic.drawing import (
    draw_blank,
    get_canvas,
    blit,
    draw,
    video_transfer,
)


class ComplexView:
    """
    Show a sound signal on screen.
    """

    def __init__(
        self, screen, size=800, antialias=True, lines=False, colours=True
    ):
        self.surface = screen
        self.size = size
        self.antialias = antialias
        self.lines = lines
        self.colours = colours
        self.img = get_canvas(size)

    def render(self, signal):
        draw_blank(self.img)
        img = draw(
            signal,
            self.size,
            antialias=self.antialias,
            lines=self.lines,
            colours=self.colours,
            axis=True,
            img=self.img,
            screen=self.surface,
        )

        if img is not None:  # Pygame drawing methods do not return img
            blit(self.surface, img)


class VideoTransferView:
    """
    Show a sound signal using the old video tape audio recording technique.
    See: http://en.wikipedia.org/wiki/44100_Hz#Recording_on_video_equipment
    """

    def __init__(self, screen, size=720, standard='PAL', axis='real'):
        self.surface = screen
        self.img = get_canvas(size)
        self.size = size
        self.standard = standard
        self.axis = axis

    def render(self, signal):
        size = self.surface.get_size()[0]
        img = draw_blank(self.img)
        tfer = video_transfer(
            signal, standard=self.standard, axis=self.axis, horiz=self.size
        )

        black = int(round((size - tfer.shape[0]) / 2.0))
        pixels = tfer[:, : img.shape[1], :].transpose(1, 0, 2)
        img[:, black:-black, :] = pixels

        blit(self.surface, img)
