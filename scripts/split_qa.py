import json
import random
from pathlib import Path

src = json.loads(Path('data/golden_qa_curated.json').read_text())
rng = random.Random(42)
groups = {}
for item in src:
    groups.setdefault(item['type'], []).append(item)
dev, test = [], []
for items in groups.values():
    rng.shuffle(items)
    n_dev = max(1, round(len(items) * 0.2))
    dev.extend(items[:n_dev])
    test.extend(items[n_dev:])
rng.shuffle(dev)
rng.shuffle(test)
Path('data/splits').mkdir(parents=True, exist_ok=True)
Path('data/splits/dev.json').write_text(json.dumps(dev, indent=2, ensure_ascii=False) + '\n')
Path('data/splits/test.json').write_text(json.dumps(test, indent=2, ensure_ascii=False) + '\n')
print(len(dev), len(test))
