"""Create an external voice profile from a recording and its exact transcript."""
import argparse
import base64
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.playground import Playground, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--transcript', type=Path, required=True, help='UTF-8 text file')
    parser.add_argument('--output', type=Path, required=True, help='New profile JSON outside Git')
    parser.add_argument('--label', default='My reference')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error('Store the profile and recordings outside the repository.')
    if args.output.exists():
        parser.error('Profile already exists; choose a new output path.')
    if args.audio.stat().st_size > 15 * 1024 * 1024:
        parser.error('Reference must be at most 15 MB.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the server validation and audio normalization without loading a model.
    app = Playground.__new__(Playground)
    app.references = args.output.parent / 'references'
    app.references.mkdir(exist_ok=True)
    saved = app.upload_reference({'audio': base64.b64encode(args.audio.read_bytes()).decode(),
        'transcript': args.transcript.read_text(encoding='utf-8-sig'), 'label': args.label})
    reference = json.loads((app.references / (saved['selection_id'] + '.json')).read_text(encoding='utf-8'))
    reference['selection_id'] = 'default'
    write_json(args.output, {'profile': 'custom-reference-full-sentence', 'reference': reference,
        'generation': {'do_sample': True, 'temperature': 0.9, 'top_k': 50, 'top_p': 1.0,
                       'max_new_tokens': 320, 'subtalker_dosample': True}})
    print(f'Voice profile saved: {args.output}')


if __name__ == '__main__':
    main()
