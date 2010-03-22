class MyClass:
    """A simple example class"""
    i = 12345
    def f(self):
        return 'hello world'

# Many ways to use properties
from recipes import prop

class Sampler:
    sr = SampleRate()
    rate = 44100

    def __new__(self):
        self.rating = 48000

    #@prop
    def rating():
        """ Sample rate"""
        return {'fget': lambda self: getattr(self, name)}
    prop(rating)
    classmethod(rating)

    @classmethod
    def _get_tuning(cls):
        print "Here"
        if type(cls.rate) == property:
            cls.rate = 44100
        return cls.rate
    @classmethod
    def _set_tuning(cls, rate):
        cls.rate = rate
    rate = property(_get_tuning, _set_tuning)