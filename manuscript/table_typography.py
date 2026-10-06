"""Math-mode numeric cells without altering their digits or precision."""
import re

NUMBER = r'-?\d+(?:\.\d+)?'
PLAIN = re.compile(NUMBER + r'(?:/' + NUMBER + r')*')

def numeric_math(cell):
    return '$' + cell + '$' if '-' in cell and PLAIN.fullmatch(cell) else cell

def plain_cell(cell):
    if cell.startswith('$') and cell.endswith('$') and PLAIN.fullmatch(cell[1:-1]):
        return cell[1:-1]
    return cell

def canonical_rows(name, rows):
    """Compare the existing numeric contracts across harmless layout changes."""
    rows = [[plain_cell(c) for c in row] for row in rows]
    if name == 'h100_pending_table.tex' and len(rows[0]) == 9:
        folded = []
        for i in range(0, 6, 2):
            r,s = rows[i:i+2]
            assert r[3] == '$R$' and s[3] == '$S$'
            assert s[:3] == ['', '', ''] and r[6] == ''
            stack = lambda j: r'\shortstack{' + r[j] + r'\\' + s[j] + '}'
            folded.append(r[:3] + [stack(4),stack(5),s[6],stack(7),stack(8)])
        for row in rows[6:]:
            assert row[3] == '---'
            folded.append(row[:3] + row[4:])
        rows = folded
    # Mean and confidence limits may occupy one line or a shortstack.
    return [[re.sub(r'\\shortstack\{\$([^$]+)\$\\\\\$([^$]+)\$\}',
                    lambda m: '$'+m[1]+r'\;'+m[2]+'$',c) for c in row] for row in rows]

if __name__ == '__main__':
    from pathlib import Path
    root = Path(__file__).resolve().parent
    visited = set()
    def walk(name):
        path = root / name
        if path in visited:
            return
        visited.add(path)
        text = path.read_text()
        for sub in re.findall(r'\\input\{([^}]+)\}', text):
            walk(sub if sub.endswith('.tex') else sub+'.tex')
    walk('main.tex')
    changes = {}
    for path in sorted(visited):
        if not path.name.endswith('_table.tex'):
            continue
        lines = path.read_text().splitlines(keepends=True)
        count = 0
        for i,line in enumerate(lines):
            if '&' not in line or not line.rstrip().endswith(r'\\'):
                continue
            body = line.rstrip()[:-2]
            cells = body.split('&')
            formatted = []
            for c in cells:
                old = c.strip()
                new = numeric_math(old)
                count += old != new
                formatted.append(new)
            if any(numeric_math(c.strip()) != c.strip() for c in cells):
                lines[i] = ' & '.join(formatted) + '\\\\\n'
        if count:
            path.write_text(''.join(lines))
            changes[path.name] = count
    print(changes)
