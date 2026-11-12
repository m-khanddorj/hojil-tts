"""Offline Qwen 0.6B full talker or LoRA training with held-out selection.

Consumes the recovered cohort and existing codec shards. All paths are explicit;
never downloads data/models, rents compute, or overwrites a historical run.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from .policy import (conditioning, external, fingerprint, learning_rate,
                     microbatches, update_groups, validate_cohorts)
from .optimization import selection_update, set_learning_rate, split_lora_optimizer
from .continuation import validate_parent
from .overfit import update_guard, validate_guard
from .observability import (adapter_movement, gradient_metrics, matched_probe,
                            summarize_losses, window_summary)

LORA_TARGETS = (r'^(model|code_predictor\.model)\.layers\.\d+\.'
                r'(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$'
                r'|^text_projection\.linear_fc[12]$')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def save(path, value):
    import torch
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def inject(talker, rank=16, dropout=0.05):
    from peft import LoraConfig, inject_adapter_in_model
    talker.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank,
                            lora_dropout=dropout, target_modules=LORA_TARGETS), talker)
    params = [p for p in talker.parameters() if p.requires_grad]
    # Low-level injection in PEFT 0.21.2 left these BF16 in the historical run.
    # Cast BEFORE creating/loading Adam so gradients and moment buffers are FP32.
    for param in params:
        param.data = param.data.float()
    return params


def configure(model, method, rank=16, dropout=0.05, gradient_checkpointing=False):
    """Full means EVERY talker parameter, including both decoders/embeddings.

    The pretrained speech codec and speaker encoder remain frozen in both modes.
    """
    import torch
    model.requires_grad_(False)
    if method == 'full':
        model.talker.to(dtype=torch.float32).requires_grad_(True)
        params = list(model.talker.parameters())
    elif method == 'lora':
        params = inject(model.talker, rank, dropout)
    else:
        raise ValueError('method must be full or lora')
    if gradient_checkpointing:
        for decoder in (model.talker.model, model.talker.code_predictor.model):
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return params


def load_adapter(talker, weights):
    from peft import set_peft_model_state_dict
    result = set_peft_model_state_dict(talker, weights)
    if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):
        raise ValueError('Adapter keys differ from the model')


def check_optimizer(optimizer):
    import torch
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dtype != torch.float32:
                raise ValueError('Trainable parameters must be FP32')
            for key in ('exp_avg', 'exp_avg_sq'):
                value = optimizer.state[param].get(key)
                if value is not None and value.dtype != torch.float32:
                    raise ValueError('Adam moments must be FP32')


def ids(wrapper, text, reference=False):
    template = wrapper._build_ref_text(text) if reference else wrapper._build_assistant_text(text)
    return wrapper.processor(text=template, return_tensors='pt')['input_ids']


def assignment(row, data, tokens, mode):
    target, ref = data[row['id']], data[row['ref_id']]
    target_ids = tokens[row['id'], False]
    ref_ids = tokens[row['ref_id'], True] if mode['icl'] else None
    # An upper bound on the actual prefix includes both transcripts and reference
    # codec frames, unlike the old target-only frame budget.
    length = 16 + target_ids.numel() + len(target['codes'])
    if ref_ids is not None:
        length += ref_ids.numel() + len(ref['codes'])
    return dict(row=row, target=target, reference=ref, target_ids=target_ids,
                ref_ids=ref_ids, mode=mode, frames=len(target['codes']), length=length)


def sequences(model, batch):
    from .sequence import build
    return [build(model, x['target_ids'], x['target']['codes'], x['reference']['xvec'],
                  x['ref_ids'], x['reference']['codes'] if x['ref_ids'] is not None else None,
                  x['mode']['non_streaming']) for x in batch]


def backward(model, assigned, frame_limit, token_limit):
    import torch
    from .sequence import loss_sums
    frames = sum(x['frames'] for x in assigned)
    first_count, residual_count = frames + len(assigned), frames * 15
    total = torch.zeros((), device='cuda', dtype=torch.float32)
    first_sum, residual_sum = total.clone(), total.clone()
    for batch in microbatches(assigned, frame_limit, token_limit):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            first, residual, _, _ = loss_sums(model, sequences(model, batch))
            loss = first / first_count + 0.3 * residual / residual_count
        loss.backward()
        total += loss.detach()
        first_sum += first.detach()
        residual_sum += residual.detach()
    if not torch.isfinite(total):
        raise ValueError('Nonfinite loss')
    return dict(loss=total.item(), first_sum=first_sum.item(), residual_sum=residual_sum.item(),
                first_tokens=first_count, residual_tokens=residual_count)


def evaluate(model, assigned, details_path=None):
    import torch
    from .sequence import loss_sums
    was_training = model.talker.training
    model.talker.eval()
    records = []
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for item in assigned:
                main, sub, n, m = loss_sums(model, sequences(model, [item]))
                row = item['row']
                records.append(dict(id=row['id'], speaker_id=row['speaker_id'],
                    group=row['validation_group'], first_sum=main.item(), residual_sum=sub.item(),
                    first_tokens=n, residual_tokens=m))
    finally:
        model.talker.train(was_training)
    result = summarize_losses(records)
    if details_path is not None:
        write(details_path, records)
    return result


def validation_features(wrapper, backup, selected, output, signature, cache=None):
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    path = output / 'validation-features.pt'
    if not path.exists() and cache is not None:
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Shared validation cache signature mismatch')
        save(path, saved)
        return saved['data']
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Validation cache signature mismatch')
        return saved['data']
    root = (backup / 'dataset-v1').resolve()
    inventory = read(root / 'dataset-inventory.json')['files']
    data = {}
    wrapper.model.speaker_encoder.to('cuda').eval()
    codec = wrapper.model.speech_tokenizer
    codec.model.to('cuda').eval()
    codec.device = torch.device('cuda:0')
    with torch.inference_mode():
        for row in selected:
            for key, name in ((row['id'], row['audio']), (row['ref_id'], row['ref_audio'])):
                if key in data:
                    continue
                audio = (root / name).resolve()
                if not audio.is_relative_to(root) or digest(audio) != inventory[name]['sha256']:
                    raise ValueError('Invalid validation audio path or checksum')
                wav, sr = sf.read(audio, dtype='float32')
                if sr != 24000 or wav.ndim != 1 or not np.isfinite(wav).all():
                    raise ValueError('Invalid validation waveform')
                codes = codec.encode(wav, sr=sr).audio_codes[0].to('cpu', torch.int16)
                mel = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024,
                    num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024,
                    fmin=0, fmax=12000).transpose(1, 2)
                encoder = wrapper.model.speaker_encoder
                vec = encoder(mel.to(device='cuda', dtype=next(encoder.parameters()).dtype))[0].to('cpu', torch.float16)
                data[key] = {'id': key, 'codes': codes, 'xvec': vec.reshape(-1)}
    wrapper.model.speaker_encoder.to('cpu')
    codec.model.to('cpu')
    codec.device = torch.device('cpu')
    torch.cuda.empty_cache()
    save(path, {'signature': signature, 'data': data})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path, help='LoRA only: fork matching adapter/optimizer; modern checkpoints also retain RNG')
    parser.add_argument('--resume', action='store_true', help='Resume this exact new run')
    parser.add_argument('--validation-cache', type=Path, help='Reuse a signature-verified feature cache')
    args = parser.parse_args()
    if args.resume and args.init_checkpoint is None and (args.output/'run.json').exists():
        if read(args.output/'run.json').get('parent') is not None:
            parser.error('Resuming a fork requires its original --init-checkpoint for provenance verification')
    cfg = read(args.config)
    method = cfg.get('method', 'lora')
    observe = cfg.get('observability', False)
    early_stopping = cfg.get('early_stopping', True)
    guard_config = cfg.get('overfit_guard')
    if guard_config:
        validate_guard(guard_config)
        if not observe:
            parser.error('Overfit guard requires matched training probes (observability=true)')
    ratio = cfg.get('loraplus_b_to_a_ratio', 1.0)
    if not math.isfinite(ratio) or ratio < 1 or (method != 'lora' and ratio != 1):
        parser.error('LoRA+ requires LoRA and a finite B/A ratio >= 1')
    if method not in ('full', 'lora') or (method == 'full' and args.init_checkpoint):
        parser.error('Full training starts from the pinned base; use --resume for an existing full run')
    steps, accum = cfg['updates'], cfg['clips_per_update']
    if (steps < 1 or accum < 1 or cfg['eval_every'] < 1 or cfg['patience'] < 1
            or cfg['min_delta'] < 0 or min(cfg['frame_limit'], cfg['token_limit']) < 1):
        parser.error('Invalid training configuration')
    learning_rate(1, steps, cfg['peak_lr'], cfg['end_lr'], cfg['warmup_updates'])
    output = external(args.output)
    if args.resume:
        if not (output / 'resume.pt').is_file():
            parser.error('No checkpoint to resume')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from peft import get_peft_model_state_dict
    from qwen_tts import Qwen3TTSModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('This trainer requires a BF16-capable CUDA GPU')
    if method == 'full' and torch.cuda.get_device_properties(0).total_memory < 22 * 2**30:
        raise ValueError('FP32 full talker + Adam requires a GPU with at least 24 GB VRAM; 48 GB recommended')
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    free, total = torch.cuda.mem_get_info()
    fraction = min(0.84, (free - 256 * 2**20) / total)
    if fraction <= 0:
        raise ValueError('Insufficient free GPU memory')
    torch.cuda.set_per_process_memory_fraction(fraction)
    signature = read(args.features / 'signature.json')
    train = rows(args.backup / 'cohorts/train.jsonl')
    recovered = read(args.backup / 'recovery-verification.json')['cohorts']
    for name in ('train', 'unseen-validation', 'familiar-validation'):
        if digest(args.backup / f'cohorts/{name}.jsonl') != recovered[name]['sha256']:
            raise ValueError(f'Changed cohort: {name}')
    if (signature['train_sha256'] != recovered['train']['sha256']
            or signature['base_revision'] != args.base.name or signature['rows'] != len(train)
            or signature['codec_sha256'] != digest(args.base / 'speech_tokenizer/model.safetensors')):
        raise ValueError('Feature/base provenance mismatch')
    manifest = args.backup / 'dataset-v1/manifest.jsonl'
    if digest(manifest) != read(args.backup / 'dataset-v1/dataset-inventory.json')['files']['manifest.jsonl']['sha256']:
        raise ValueError('Changed dataset manifest')
    unseen = rows(args.backup / 'cohorts/unseen-validation.jsonl')
    familiar = defaultdict(list)
    for row in rows(args.backup / 'cohorts/familiar-validation.jsonl'):
        if row.get('eligible'):
            familiar[row['speaker_id']].append(row)
    selected = [dict(row, validation_group='unseen') for row in unseen]
    selected += [dict(min(group, key=lambda r: fingerprint([cfg['seed'], r['id']])),
                      validation_group='familiar') for _, group in sorted(familiar.items())]
    canonical = validate_cohorts(train, selected, rows(manifest))
    if {r['speaker_id'] for r in train} & {r['speaker_id'] for r in unseen}:
        raise ValueError('Unseen validation speakers appear in training')
    shard_hashes = {p.name: digest(p) for p in sorted(args.features.glob('*.pt'))}
    provenance = {'config': cfg, 'features': signature, 'shards': shard_hashes,
                  'validation': fingerprint(selected), 'base_config': digest(args.base / 'config.json'),
                  'base_weights': digest(args.base / 'model.safetensors'),
                  'source': {name: digest(Path(__file__).parent / name)
                             for name in ('engine.py', 'sequence.py', 'policy.py', 'optimization.py',
                                          'observability.py', 'monitor.py', 'continuation.py', 'overfit.py')},
                  'parent': digest(args.init_checkpoint) if args.init_checkpoint else None,
                  'versions': {name: importlib.metadata.version(name)
                               for name in ('torch', 'qwen-tts', 'peft', 'transformers')},
                  'gpu': torch.cuda.get_device_name(0), 'kind': method + '-fp32-parameters',
                  'validation_clips': len(selected)}
    if args.resume and read(output / 'run.json') != provenance:
        raise ValueError('Resume requires identical data, model, source and configuration')
    data = {}
    for name in shard_hashes:
        for item in torch.load(args.features / name, map_location='cpu', weights_only=True):
            if item['id'] in data:
                raise ValueError('Duplicate feature ID')
            data[item['id']] = item
    if set(data) != {r['id'] for r in train}:
        raise ValueError('Feature shards differ from training IDs')
    wrapper = Qwen3TTSModel.from_pretrained(str(args.base), device_map='cpu',
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    model = wrapper.model
    model.requires_grad_(False)
    data.update(validation_features(wrapper, args.backup, selected, output,
                                   fingerprint([provenance['validation'], signature]), args.validation_cache))
    for item in data.values():
        codes, vec = item['codes'], item['xvec']
        if (codes.ndim != 2 or codes.shape[1] != 16 or not codes.numel()
                or codes.min() < 0 or codes.max() >= 2048 or vec.shape != (1024,)
                or not vec.isfinite().all() or vec.norm() == 0):
            raise ValueError('Malformed cached feature')
    tokens = {}
    for row in train + selected:
        for key, reference in ((row['id'], False), (row['ref_id'], True)):
            if (key, reference) not in tokens:
                tokens[key, reference] = ids(wrapper, canonical[key]['text'], reference)
    print(f'Cached transcripts; validation covers {len(selected)} clips.', flush=True)
    params = configure(model, method, cfg.get('rank', 16), cfg.get('dropout', 0.05),
                       cfg.get('gradient_checkpointing', False))
    talker = model.talker.to('cuda')
    # Resolve parameters after device placement, before constructing the optimizer.
    params = [p for p in talker.parameters() if p.requires_grad]
    print(json.dumps({'method': method, 'trainable_parameters': sum(p.numel() for p in params),
                      'parameter_dtype': 'float32'}), flush=True)
    talker.train()
    optimizer = torch.optim.AdamW(params, lr=cfg['peak_lr'], weight_decay=0.01, fused=True)
    eval_items = [assignment(r, data, tokens, {'icl': True, 'non_streaming': True}) for r in selected]
    probe_rows = matched_probe(train, selected, data, cfg['seed']) if observe else []
    probe_items = [assignment(r, data, tokens, {'icl': True, 'non_streaming': True}) for r in probe_rows]
    if observe:
        write(output / 'train-probe.json', dict(rows=probe_rows, fingerprint=fingerprint(probe_rows),
              conditioning={'icl': True, 'non_streaming': True}, dropout=False,
              selection='One training clip per familiar validation speaker, matched by target codec length'))
    groups = []
    epoch = cfg['start_epoch']
    while len(groups) < steps:
        order = list(train)
        random.Random(cfg['seed'] + epoch).shuffle(order)
        groups.extend((epoch, group) for group in update_groups(order, accum))
        epoch += 1
    start, best, bad, best_step = 0, float('inf'), 0, 0
    patience_anchor = float('inf')
    clips_seen, codec_frames_seen, oom_retries = 0, 0, 0
    inherited_updates, inherited_clips = 0, 0
    guard_state = None
    optimizer_grouped = False
    frame_limit, token_limit = cfg['frame_limit'], cfg['token_limit']
    parent = output / 'resume.pt' if args.resume else args.init_checkpoint
    if parent:
        state = torch.load(parent, map_location='cpu', weights_only=True)
        modern_fork = validate_parent(state, provenance) if not args.resume else False
        if method == 'full':
            talker.load_state_dict(state['talker'], strict=True)
        else:
            load_adapter(talker, state['adapter'])
        if args.resume or modern_fork:
            split_lora_optimizer(optimizer, talker.named_parameters(), ratio)
            optimizer_grouped = True
        optimizer.load_state_dict(state['optimizer'])
        if guard_config and (args.resume or modern_fork):
            previous_guard = state['provenance']['config'].get('overfit_guard')
            if previous_guard == guard_config:
                if state.get('guard_probe_fingerprint') != fingerprint(probe_rows):
                    raise ValueError('Overfit guard requires the same matched training probe')
                guard_state = state.get('overfit_guard')
        if args.resume:
            if state['provenance'] != provenance:
                raise ValueError('Checkpoint provenance mismatch')
            start, best, bad, best_step = state['step'], state['best'], state['bad'], state['best_step']
            patience_anchor = state['patience_anchor']
            clips_seen = state.get('clips_seen', sum(len(g) for _, g in groups[:start]))
            codec_frames_seen = state.get('codec_frames_seen', 0)
            oom_retries = state.get('oom_retries', 0)
            inherited_updates = state.get('inherited_updates', 0)
            inherited_clips = state.get('inherited_clips', 0)
            frame_limit, token_limit = state['frame_limit'], state['token_limit']
            torch.set_rng_state(state['rng_cpu'])
            torch.cuda.set_rng_state(state['rng_cuda'])
        elif modern_fork:
            inherited_updates = state.get('total_updates',state['step'])
            inherited_clips = state.get('total_clips_seen',state.get('clips_seen',0))
            torch.set_rng_state(state['rng_cpu'])
            torch.cuda.set_rng_state(state['rng_cuda'])
        del state
    if not args.resume and not optimizer_grouped:
        split_lora_optimizer(optimizer, talker.named_parameters(), ratio)
    check_optimizer(optimizer)
    write(output / 'run.json', provenance)
    source = output / 'source'
    source.mkdir(exist_ok=True)
    for name in provenance['source']:
        shutil.copy2(Path(__file__).parent / name, source / name)

    def trainable_state():
        state = talker.state_dict() if method == 'full' else get_peft_model_state_dict(talker)
        return {k: v.detach().cpu() for k, v in state.items()}

    last_probe_weights = trainable_state() if observe and method == 'lora' else None

    def checkpoint(step, path=None):
        save(path or output / 'resume.pt', dict(provenance=provenance, step=step,
             **{('talker' if method == 'full' else 'adapter'): trainable_state()},
             optimizer=optimizer.state_dict(), best=best, bad=bad, best_step=best_step,
             patience_anchor=patience_anchor,
             overfit_guard=guard_state,
             guard_probe_fingerprint=fingerprint(probe_rows) if guard_config else None,
             clips_seen=clips_seen, codec_frames_seen=codec_frames_seen, oom_retries=oom_retries,
             inherited_updates=inherited_updates, inherited_clips=inherited_clips,
             total_updates=inherited_updates+step, total_clips_seen=inherited_clips+clips_seen,
             frame_limit=frame_limit, token_limit=token_limit,
             rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()))

    def select(step):
        nonlocal best, bad, best_step, patience_anchor, last_probe_weights, guard_state
        evaluation_started = time.monotonic()
        metrics = evaluate(model, eval_items, output / f'validation-cases-{step:06d}.json' if observe else None)
        probe_metrics, movement = None, None
        if observe:
            probe_metrics = evaluate(model, probe_items, output / f'train-probe-cases-{step:06d}.json')
            if last_probe_weights is not None:
                current_weights = trainable_state()
                movement = adapter_movement(last_probe_weights, current_weights)
                last_probe_weights = current_weights
        if guard_config:
            guard_state = update_guard(guard_state, step=inherited_updates+step,
                train=probe_metrics['all']['combined'], familiar=metrics['familiar']['combined'],
                unseen=metrics['unseen']['combined'], config=guard_config)
        best, patience_anchor, bad, improved = selection_update(
            metrics['selection_loss'], best, patience_anchor, bad, cfg['min_delta'])
        if improved:
            best_step = step
            best_path = output / f"best-{'talker' if method == 'full' else 'adapter'}-{step:06d}.pt"
            save(best_path, trainable_state())
            resume_path = output / f'best-checkpoint-{step:06d}.pt'
            checkpoint(step, resume_path)
            write(output / 'best.json', {'step': step, 'metrics': metrics,
                                       'checkpoint': resume_path.name, 'checkpoint_sha256': digest(resume_path),
                                       'weights': best_path.name, 'weights_sha256': digest(best_path)})
        event = dict(step=step, metrics=metrics, improved=improved, bad_checks=bad,
                     total_updates=inherited_updates+step, total_clips_seen=inherited_clips+clips_seen,
                     patience_anchor=patience_anchor, best_step=best_step, train_probe=probe_metrics,
                     adapter_movement=movement, clips_seen=clips_seen, codec_frames_seen=codec_frames_seen,
                     epoch_fraction=clips_seen/len(train), evaluation_seconds=time.monotonic()-evaluation_started)
        event['overfit_guard'] = guard_state
        if probe_metrics is not None:
            event['familiar_generalization_gap'] = metrics['familiar']['combined'] - probe_metrics['all']['combined']
            event['unseen_generalization_gap'] = metrics['unseen']['combined'] - probe_metrics['all']['combined']
        write(output / f'validation-{step:06d}.json', event)
        print(json.dumps({'validation_step': step, 'selection_loss': metrics['selection_loss'],
              'familiar_loss': metrics['familiar']['combined'], 'unseen_loss': metrics['unseen']['combined'],
              'train_probe_loss': probe_metrics['all']['combined'] if probe_metrics else None,
              'best_step': best_step, 'bad_checks': bad}), flush=True)
        checkpoint(step)
        if observe:
            from .monitor import render
            render(output)

    if not args.resume:
        select(0)  # The unmodified parent is eligible to win.
    already_stopped = ((early_stopping and bad >= cfg['patience'])
                       or bool(guard_state and guard_state['stopped']))
    started = time.monotonic()
    window = []
    step = start
    for index in range(start, start if already_stopped else steps):
        step = index + 1
        update_started = time.monotonic()
        epoch, group = groups[index]
        assigned = [assignment(r, data, tokens,
                    conditioning(r['id'], cfg['seed'], epoch, cfg['non_streaming_only'])) for r in group]
        rate = learning_rate(step, steps, cfg['peak_lr'], cfg['end_lr'], cfg['warmup_updates'])
        set_learning_rate(optimizer, rate)
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                losses = backward(model, assigned, frame_limit, token_limit)
            except torch.OutOfMemoryError:
                oom_retries += 1
                if frame_limit == 1 and token_limit == 1:
                    raise
                frame_limit, token_limit = max(1, frame_limit // 2), max(1, token_limit // 2)
            else:
                break
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(gpu_rng)
            print(f'OOM: persist smaller budgets {frame_limit}/{token_limit}', flush=True)
        log_step = step == 1 or step % 10 == 0 or step == steps
        component_gradients = gradient_metrics(talker.named_parameters()) if observe and log_step else None
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        # Never catch OOM around optimizer.step: a partially updated optimizer
        # cannot safely be retried. Resume from the last durable checkpoint.
        optimizer.step()
        check_optimizer(optimizer)
        clips_seen += len(group)
        frames_in_update = sum(x['frames'] for x in assigned)
        codec_frames_seen += frames_in_update
        window.append(dict(**losses, gradient_norm=norm.item(), clips=len(group),
                           codec_frames=frames_in_update, step_seconds=time.monotonic()-update_started))
        if log_step:
            event = dict(step=step, **losses, gradient_norm=norm.item(), lr=rate,
                         total_updates=inherited_updates+step, total_clips_seen=inherited_clips+clips_seen,
                         group_lrs=[g['lr'] for g in optimizer.param_groups],
                         component_gradients=component_gradients, clips_seen=clips_seen,
                         codec_frames_seen=codec_frames_seen, epoch_fraction=clips_seen/len(train),
                         oom_retries=oom_retries, **window_summary(window),
                         seconds=time.monotonic()-started, frames=frame_limit, tokens=token_limit,
                         peak_reserved=torch.cuda.max_memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated())
            write(output / f'metrics-{step:06d}.json', event)
            print(json.dumps(event), flush=True)
            window.clear()
        epoch_end = index + 1 == len(groups) or groups[index+1][0] != epoch
        if step % cfg['eval_every'] == 0 or step == steps or (cfg.get('save_epochs') and epoch_end):
            select(step)
            if cfg.get('save_epochs') and epoch_end:
                checkpoint(step, output / f'epoch-{epoch+1:03d}-checkpoint.pt')
            if (early_stopping and bad >= cfg['patience']) or (guard_state and guard_state['stopped']):
                break
    stop_reason = ('overfit_guard' if guard_state and guard_state['stopped'] else
                   'validation_patience' if early_stopping and bad >= cfg['patience'] else 'update_limit')
    write(output / 'result.json', {'completed_updates': step, 'best_step': best_step,
          'best_selection_loss': best, 'early_stopped': step < steps and stop_reason != 'update_limit',
          'stop_reason': stop_reason, 'overfit_guard': guard_state,
          'clips_seen': clips_seen, 'epoch_fraction': clips_seen/len(train), 'oom_retries': oom_retries,
          'total_updates': inherited_updates+step, 'total_clips_seen': inherited_clips+clips_seen,
          'requires_generated_speech_evaluation': True})
    if observe:
        from .monitor import render
        render(output)


if __name__ == '__main__':
    main()
