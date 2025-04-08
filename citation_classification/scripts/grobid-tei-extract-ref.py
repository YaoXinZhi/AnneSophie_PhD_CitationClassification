import xml.etree.ElementTree as ET
import sys
import csv
import os.path
import collections


NS = dict(
    tei='http://www.tei-c.org/ns/1.0',
    xml='http://www.w3.org/XML/1998/namespace'
    )


def sentence_before(sent, ref):
    for e in sent.iter():
        if e == ref:
            break
        if e.text is not None:
            yield e.text
        if e.tail is not None:
            yield e.tail


def sentence_after(e, ref):
    if ref is None and e.text is not None:
        yield e.text
    for c in e:
        if c == ref:
            ref = None
            if c.tail is not None:
                yield c.tail
            continue
        yield from sentence_after(c, ref)
    if ref is None and e.tail is not None:
        yield e.tail


def text(x):
    if x is None:
        return ''
    if isinstance(x, ET.Element):
        return text(x.itertext())
    return ''.join(x)


def xmlid(e):
    return e.attrib['{%s}id' % NS['xml']]


def get_sections(e, sec_name=None):
    for head in e.findall('tei:head', NS):
        if head.text != '':
            sec_name = head.text
            break
    yield e, sec_name
    for c in e:
        yield from get_sections(c, sec_name)


def ref_rows(fn, sentence_context):
    tree = ET.parse(fn)
    root = tree.getroot()
    sections = dict(get_sections(root))
    nref_sections = collections.Counter(sections[ref] for ref in root.findall('.//tei:ref[@type = "bibr"]', NS))
    bib = dict(('#' + xmlid(b), b) for b in root.findall('.//tei:listBibl/tei:biblStruct', NS))
    row_common = [os.path.basename(fn)]
    for div in root.findall('.//*[tei:s]', NS):
        sentences = div.findall('tei:s', NS)
        padded = [None] * sentence_context + sentences + [None] * sentence_context
        for n, sent in enumerate(sentences):
            sec_name = sections[sent]
            m = n + sentence_context
            refs = sent.findall('tei:ref[@type = "bibr"]', NS)
            for ref in refs:
                row = list(row_common)
                row.append(sec_name)
                row.append(xmlid(sent))
                row.extend(text(padded[m - i]) for i in range(sentence_context, 0, -1))
                row.append(text(sentence_before(sent, ref)))
                row.append(text(ref))
                row.append(text(sentence_after(sent, ref)))
                row.extend(text(padded[m + i + 1]) for i in range(sentence_context))
                row.append(len(refs))
                row.append(nref_sections[sec_name])
                row.append(bib.get(ref.attrib.get('target', ''), ref).findtext('.//tei:note', '', NS))
                row.append(bib.get(ref.attrib.get('target', ''), ref).findtext('.//tei:idno[@type = "DOI"]', '', NS))
                yield row


def header_row(sentence_context):
    yield 'FILE'
    yield 'SECTION'
    yield 'SENT_ID'
    for i in range(sentence_context, 0, -1):
        yield f'S-{i}'
    yield 'S-0'
    yield 'CITANCE'
    yield 'S+0'
    for i in range(sentence_context):
        yield f'S+{i + 1}'
    yield '#REFS/SENT'
    yield '#REFS/SECTION'
    yield 'FULL_REF'
    yield 'REF_DOI'


w = csv.writer(sys.stdout)
w.writerow(header_row(3))
for fn in sys.argv[1:]:
    w.writerows(ref_rows(fn, 3))
