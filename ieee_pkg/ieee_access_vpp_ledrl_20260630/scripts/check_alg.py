import re
from collections import Counter

src = open('main.tex', encoding='utf-8').read()
blocks = list(re.finditer(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', src, re.S))
print('found', len(blocks), 'algorithm blocks')
for i, m in enumerate(blocks):
    blk = m.group(0)
    begs = Counter(re.findall(r'\\begin\{(\w+)\}', blk))
    ends = Counter(re.findall(r'\\end\{(\w+)\}', blk))
    print(f'--- block {i+1} ---')
    print('  begin:', dict(begs))
    print('  end  :', dict(ends))
    for a, b in [('For', 'EndFor'), ('If', 'EndIf'), ('Function', 'EndFunction'), ('While', 'EndWhile')]:
        pat_b = r'\\' + a + r'(?![a-zA-Z])'
        pat_e = r'\\' + b + r'(?![a-zA-Z])'
        nb = len(re.findall(pat_b, blk))
        ne = len(re.findall(pat_e, blk))
        print(f'  {a}/{b}: {nb} / {ne}  {"OK" if nb == ne else "MISMATCH"}')
    # any stray bare $ that would break math (should be paired)
    dollar = blk.count('$')
    print('  $ count (should be even):', dollar)
