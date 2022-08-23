from __future__ import unicode_literals
import regex
from itertools import imap
from pandas import read_csv

dat = read_csv('/home/datarobot/workspace/data-science-scripts/zach/lc363.csv')
a = dat.iloc[361,][['desc']].values[0]
a = a.decode('utf8')

dat = read_csv('/home/datarobot/workspace/data-science-scripts/zach/lc900.csv')
b = dat.iloc[361,][['desc']].values[0]
b = b.decode('utf8')

assert a == b

r1 = regex.compile("[^\p{L}\p{M}\p{N}\p{P}\p{S}\p{Z}]+")
r1.sub("", a)

for phrase in [
    "の, は, でした",
     "コンサート",
    " 昨夜, 最高",
    " おはようございます。",
    unichr(0x2603),
    "Oui, ça va, et toi ?"]:
    assert repr(r1.sub("", phrase)) == repr(phrase)

a_full = dat[['desc']].values

strip_black_magic_unicode_chars = regex.compile("[^\p{L}\p{M}\p{N}\p{P}\p{S}\p{Z}]+")
def unicode_strip_magic(x):
    return strip_black_magic_unicode_chars.sub("", unicode(x))

for i in imap(unicode_strip_magic, a_full):
    print i
