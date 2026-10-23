import argparse
import json
import sys
from pathlib import Path

from .adaptation import export_adaptation
from .io import read_rows
from .pipeline import prepare
from .qwen import encode, export, validate_codes
from .sources import acquire, import_parquet


def main(argv=None):
    parser = argparse.ArgumentParser(description='Khalkha speech corpus preparation (no implicit downloads)')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('acquire', 'prepare', 'import-mbspeech'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--output', required=True)
        if name != 'prepare':
            p.add_argument('--source', required=True)
        if name == 'import-mbspeech':
            p.add_argument('parquet', nargs='+')
        if name == 'prepare':
            p.add_argument('--workers', type=int, default=1)
            p.add_argument('--resume', action='store_true')
    p = sub.add_parser('export-adaptation')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('export-qwen')
    p.add_argument('--manifest', required=True)
    p.add_argument('--speaker', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('encode-qwen')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tokenizer-model', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    p = sub.add_parser('check-codes')
    p.add_argument('--input', required=True)
    p = sub.add_parser('audit-transcripts')
    p.add_argument('--manifest', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--cer-threshold', type=float, default=.3)
    p.add_argument('--backend', choices=['faster-whisper', 'whisper-cpp'], default='faster-whisper')
    p.add_argument('--executable')
    args = parser.parse_args(argv)
    try:
        if hasattr(args, 'config'):
            cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
            if args.command != 'prepare':
                matches = [s for s in cfg['sources'] if s['id'] == args.source]
                if len(matches) != 1:
                    raise ValueError('Select exactly one configured source')
                source = matches[0]
        if args.command == 'acquire':
            acquire(source, args.output)
        elif args.command == 'prepare':
            if args.workers != 1 or args.resume:
                from .bulk import prepare_bulk
                report = prepare_bulk(cfg, args.output, args.workers, args.resume)
            else:
                report = prepare(cfg, args.output)
            print(json.dumps({k: report[k] for k in ('accepted', 'rejected', 'accepted_hours', 'reasons')}, indent=2))
        elif args.command == 'import-mbspeech':
            import_parquet(source, args.parquet, args.output)
        elif args.command == 'export-adaptation':
            print(json.dumps(export_adaptation(args.manifest, args.output)))
        elif args.command == 'export-qwen':
            export(args.manifest, args.speaker, args.output)
        elif args.command == 'encode-qwen':
            encode(args.input, args.output, args.tokenizer_model, args.device, args.batch_size)
        elif args.command == 'audit-transcripts':
            from .transcripts import audit_transcripts
            report = audit_transcripts(args.manifest, args.model_dir, args.output, args.device, args.cer_threshold,
                                       args.backend, args.executable)
            print(json.dumps({k: report[k] for k in ('rows', 'flagged', 'cer_threshold')}))
        else:
            count = 0
            for row in read_rows(args.input):
                validate_codes(row)
                count += 1
            if not count:
                raise ValueError('Empty training input')
            print(f'Validated {count} coded rows')
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')
