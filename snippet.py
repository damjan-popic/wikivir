import xml.etree.ElementTree as ET, sys
fname = "normalized_corpus.xml"
hits = 0
for ev, el in ET.iterparse(fname, events=("end",)):
    if el.tag == "doc":
        hits += 1
        if hits <= 3:                         # show first few for sanity
            print("TAG:", el.tag, "TEXT len:", len(el.text or ""))
    el.clear()
print("TOTAL <doc> elements:", hits)
