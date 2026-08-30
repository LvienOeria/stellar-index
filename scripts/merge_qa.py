import json
from pathlib import Path

base = json.loads(Path('data/golden_qa.json').read_text())
extra = json.loads(Path('/tmp/qa-extra.json').read_text())
seen = {q['question'] for q in base}
for item in extra:
    if item['question'] not in seen:
        seen.add(item['question'])
        base.append(item)
for i, item in enumerate(base):
    item.setdefault('id', f"{item.get('book_id','book')}-{i:03d}")
Path('data/golden_qa.json').write_text(json.dumps(base, indent=2, ensure_ascii=False) + '\n')
print(len(base))
