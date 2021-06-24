def polygon_osc(n=6, curve=Super(6, 1.5, 1.5, 1.5), harmonics=1, rand_phase=False):
    o = Osc(sampler.rate / n, curve=curve)
    h = Harmonics(o, n=harmonics, rand_phase=rand_phase)

    return h


def test_clothoids(snd, n=6, simple=False):
    deg = pi2
    quarter = pi2 / 4
    indices = np.arange(-1, n + 1)
    points = snd[indices]

    if simple:
        # Works for circles and super ellipses
        tangents = (deg * (np.arange(-1, n + 1) / n + 0.25)) % deg
    else:
        mids = np.array([np.angle(points[i] - midpoint(points[i - 1], points[i + 1])) for i in np.arange(n)])
        tangents = (mids + quarter) % pi2
        print('tangents:', tangents / pi2)

    clothoid_list = [
        Clothoid.G1Hermite(
            points[i].real,
            points[i].imag,
            tangents[i],
            points[i + 1].real,
            points[i + 1].imag,
            tangents[i + 1],
        )
        for i in np.arange(n - 1)
    ]

    return clothoid_list


def plot_clothoids_test(n=6, simple=False, debug=False, **kwargs):
    osc = polygon_osc(n, **kwargs)
    env = Exponential(-0.987, amp=0.9)
    snd = Mix(osc, env)

    if debug:
        plot_signal(snd[:n + 1])

    clothoid_list = test_clothoids(snd, n, simple)
    for i in clothoid_list:
        plt.plot( *i.SampleXY(500) )
