"""
Wikipedia API client module.
"""

import funcy
import locale
import logging
import numpy as np
import os
import re

from wikitools import wiki, api
from fractions import Fraction

from akasha.utils.log import logger
from akasha.math import cents, identity


def user_agent(req):
    """
    Return the user agent string.
    """
    product = 'Akasha Resonance'
    url = 'https://composed.nu/peterhil/'
    platform = '; '.join(np.array(os.uname())[[0, 2, 4]])
    system = '(' + platform + '; ' + funcy.first(locale.getlocale()) + ')'
    browser_platform = req.headers['User-agent'][0]
    python_version = 'Python/' + funcy.first(os.sys.version.split(' '))

    return (
        browser_platform,
        system,
        product + ' (' + url + ')',
        python_version,
    )


def get_interval_list():
    """
    Get the list of musical pitch intervals table from Wikipedia.
    """
    site = wiki.Wiki("https://en.wikipedia.org/w/api.php")
    params = {
        'action': 'query',
        'prop': 'revisions',
        'titles': 'List of pitch intervals',
        'rvprop': 'content',
        'rvsection': '1',
        'format': 'json',
    }
    req = api.APIRequest(site, params)
    req.headers['User-agent'] = user_agent(req)  # Customize User Agent
    res = req.query(querycontinue=True)
    return res


def replacement(groups, template='', func=identity):
    """Apply a function to matched regexp and format the results
    using a template.
    """

    def repl(match):
        # pylint: disable=C0111
        if isinstance(groups, (list, tuple)):
            return template.format(*func(match.group(*groups)))
        else:
            return template.format(func(match.group(groups)))

    return repl


def remove_wiki_links(string):
    """
    Remove links in wiki syntax.
    """
    wiki_link_tag = re.compile(r"\[\[([^\|]+\|)?(.*?)\]\]")

    def repl(m):
        # pylint: disable=C0111
        return m.group(2)

    return re.sub(wiki_link_tag, repl, string)


def filter_tags(string, tag, template='', func=identity):
    """
    Filter out html tags and change their values by calling replacement()
    with a function and a template.
    """
    regex = r"<{0}( name=\"[^\"]+\")?(/>|>([^<]*?)</{0}>)"
    wiki_tag = re.compile(regex.format(tag))  # pylint: disable=C0209
    return re.sub(wiki_tag, replacement(3, template, func), string)


def remove_templates(string, tags=None):
    """Replaces all template tags (or just named tags) of the type
    '{{tag|value|other}}' with 'value|other' from the string.
    """
    wiki_template = re.compile(r"{{([A-Za-z]+)\|(.*?)}}")

    def repl(m):
        # pylint: disable=C0111
        if tags and m.group(1) not in tags:
            return m.group(0)  # Do not replace
        return m.group(2)

    return re.sub(wiki_template, repl, string) or ''


def template_items(string):
    """
    Find wiki syntax templates from a string.
    """
    wiki_template = re.compile(r"{{([A-Za-z]+)\|(.*)}}")
    res = re.findall(wiki_template, string)
    return funcy.first(res) if res else ('', '')


def template_value(string):
    """
    Get values from wiki syntax templates.
    """
    return template_items(string)[-1]


def parse_interval_name(string, only_first=False):
    """
    Parse interval name as plain text from a string.
    """
    try:
        value = template_value(string)
        name = funcy.first(value.split('|'))
        if only_first:
            name = funcy.first(funcy.first(name.split(' or ')).split(','))
        audio = template_value(value).split('|')
        try:
            audio.remove('help=no')
        except ValueError:
            pass
        return (name, funcy.first(audio))
    except KeyError:
        logger.error("KeyError for string: %s", string)
        return (name, "")


def parse_freq_ratio(string):
    """Parse frequency ratio from the expression in the factors column
    of the table.

    Example
    -------
    >>> parse_freq_ratio("2<sup>7</sup> : 5.0<sup>3</sup>")
    u'2 ** Fraction(7, 1) / 5.0 ** Fraction(3, 1)'
    """
    re_div = re.compile(r" ?(:|÷|\xf7) ?", re.UNICODE)
    re_mul = re.compile(r" ?(·|\xc2\xb7|\xb7|&middot;) ?", re.UNICODE)
    out = remove_templates(string)
    out = re.sub(re_mul, ' * ', out, re.UNICODE)
    out = re.sub(re_div, ' / ', out, re.UNICODE)
    out = filter_tags(out, tag='sup', template=' ** {0!r}', func=Fraction)
    return str(out.encode('iso-8859-1'))


def parse_wiki(res, loglevel=logging.ANIMA):
    """Parse the interval dictionary from the fetched wikipedia page
    in the wiki syntax.
    """
    pgs = res['query']['pages']
    content = pgs[pgs.keys()[0]]['revisions'][0]['*']
    links_removed = remove_wiki_links(filter_tags(content, 'ref'))
    table = links_removed.split('|+')[1].split('\n|-\n|')[:-1]
    legend, table = funcy.first(table), funcy.rest(table)
    out = []
    for row in table:
        rec = [k.strip() for k in row.split('||')]
        # Cents
        rec[0] = float(template_value(rec[0]).strip())
        # Musical note
        rec[1] = remove_templates(rec[1], 'music')
        # Interval name and audio
        if len(rec) > 5:
            rec[4:5] = parse_interval_name(rec[4], only_first=True)
        # Combine limits
        rec[-12:] = [rec[-12:]]
        # Factors
        rec[3] = parse_freq_ratio(rec[3])
        print("Factors:", rec[3])
        # Freq. ratio
        if rec[2] != 0:
            rec[2] = parse_freq_ratio(rec[2])
            cts = float(cents(eval(compile(rec[2], __name__, 'eval'))))
            if cts != 0 or rec[0] == 0:
                logger.log(loglevel, "Parsed cents: %s", cts)
                rec[0] = cts
                err = cts - rec[0]
                if np.abs(err) >= 0.005:
                    logger.warning(
                        "%s '%s' has a too big error %.5f with it's "
                        "cents %.5f != %.5f",
                        rec[2],
                        rec[1],
                        err,
                        rec[0],
                        cts,
                    )
            else:
                logger.warning(
                    "Parsed cents is %s for record:\n\t\t%s",
                    cts,
                    rec,
                )
        out.append(rec)
    return (out, legend)
