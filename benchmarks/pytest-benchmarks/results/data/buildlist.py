import sys
import json
from pathlib import Path

def main():
    benchmarkresults = []
    for path in Path('.').glob('*/*.json'):
        benchmarkresults.append({
            'name':'{}'.format(*path.parts),
            'file':str(path)
        })
    json.dump(benchmarkresults, sys.stdout)


if __name__ == "__main__":
    main()