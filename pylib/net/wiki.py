#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import locale
import numpy as np
import os
import re

from funct.xoltar.functional import car, cdr
from utils.math import identity
from wikitools import wiki, api

def user_agent(req):
    product = 'Akasha-Resonance'
    url = 'http://composed.nu/peterhil/'
    system = '(' + ('; '.join(np.array(os.uname())[[0,2,4]])) + '; ' + car(locale.getlocale()) + ')'
    browser_platform = req.headers['User-agent'][0]
    python_version = 'Python/' + car(os.sys.version.split(' '))
    return (browser_platform, system, product + ' (' + url + ')', python_version)

def get_interval_list():
    site = wiki.Wiki("http://en.wikipedia.org/w/api.php")
    params = {
        'action': 'query',
        'prop': 'revisions',
        'titles': 'List of pitch intervals',
        'rvprop': 'content',
        'rvsection': '1',
        'format': 'json'
    }
    req = api.APIRequest(site, params)
    req.headers['User-agent'] = user_agent(req) # Customize User Agent
    res = req.query(querycontinue=True)
    return res

def remove_wiki_links(string):
    wiki_link_tag = re.compile(r"\[\[([^\|]+\|)?(.*?)\]\]")
    def repl(m):
        return m.group(2)
    return re.sub(wiki_link_tag, replacement(3, ' ** {0} '), string)

def filter_tags(string, tag, template=u'', filter=identity):
    wiki_tag = re.compile(r"<{0}( name=\"[^\"]+\")?(/>|>([^<]*?)</{0}>)".format(tag))
    def repl(m):
        return template.format(filter(m.group(3))) if template else u''
    return re.sub(wiki_tag, repl, string)

def remove_template_tags(string, tags=None):
    """
    Replaces all template tags (or just named tags) of the type '{{tag|value|other}}' with 'value|other' from the string.
    """
    wiki_template = re.compile(r"{{([A-Za-z]+)\|(.*?)}}")
    def repl(m):
        if tags and not m.group(1) in tags:
            return m.group(0) # Do not replace
        return m.group(2)
    res = re.sub(wiki_template, repl, string)
    return res if res else u''

def template_items(string):
    wiki_template = re.compile(r"{{([A-Za-z]+)\|(.*)}}")
    res = re.findall(wiki_template, string)
    return car(res) if res else (u'', u'')

def template_value(string):
    return template_items(string)[-1]

def parse_interval_name(string, only_first=False):
    try:
        value = template_value(string)
        name = car(value.split('|'))
        if only_first:
            name = car(car(name.split(' or ')).split(','))
        audio = template_value(value).split('|')
        try:
            audio.remove('help=no')
        except ValueError:
            pass
        return (name, car(audio))
    except KeyError, e:
        logger.error("KeyError for string: %s" % string)
        return (name, "")

def parse_freq_ratio(string):
    out = filter_tags(string, tag='sup', template=' ** {0}', filter=lambda x: repr(Fraction(x)))
    return out

def parse_wiki(res):
    pgs = res['query']['pages']
    content = pgs[pgs.keys()[0]]['revisions'][0]['*']
    table = remove_wiki_links(filter_tags(content, 'ref')).split('|+')[1].split('\n|-\n|')[:-1]
    legend, table = car(table), cdr(table)
    out = []
    for row in table:
        rec = [k.strip() for k in row.split('||')]
        # Cents
        rec[0] = float(template_value(rec[0]).strip())
        # Musical note
        rec[1] = remove_template_tags(rec[1], 'music')
        # Interval name and audio
        if len(rec) > 5:
            rec[4:5] = parse_interval_name(rec[4], only_first=True)
        # Combine limits
        rec[-12:] = [rec[-12:]]
        out.append(rec)
    return (out, legend)

