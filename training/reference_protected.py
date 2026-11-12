"""Target-text-only LoRA experiment, with unchanged historical pilot helpers.

The adapter changes target transcript projections inside full ICL only. Reference
transcript, role/control tokens and recurrent TTS padding use frozen base layers.
This is conditioning protection, not a guarantee of perceptual voice retention.
"""
import argparse
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

from . import targeted_voice as pilot

ORIGINAL_LOAD = pilot.load_model
ORIGINAL_HASHES = pilot.source_hashes
ORIGINAL_LOAD_ADAPTER = pilot.load_adapter


def protect_reference(model):
    import torch
    projection = model.talker.text_projection
    if hasattr(projection, '_target_route'):
        raise ValueError('Routing already installed')
    for layer in (projection.linear_fc1, projection.linear_fc2):
        if not hasattr(layer, 'base_layer') or layer.merged:
            raise ValueError('Requires separate, unmerged LoRA layers')
    state = {'span': None}
    projection._target_route = state
    original_forward = projection.forward
    original_icl = model.generate_icl_prompt
    original_generate = model.generate
    signature = inspect.signature(original_icl)
    generate_signature = inspect.signature(original_generate)

    def frozen(x):
        return projection.linear_fc2.base_layer(projection.act_fn(projection.linear_fc1.base_layer(x)))

    def routed(self, hidden_state):
        baseline = frozen(hidden_state)
        if state['span'] is None:
            return baseline
        start, stop = state['span']
        if hidden_state.ndim != 3 or hidden_state.shape[1] != stop:
            raise ValueError('Unexpected ICL text projection layout')
        positions = torch.arange(stop, device=hidden_state.device)
        mask = ((positions >= start) & (positions < stop))[None, :, None]
        return torch.where(mask, original_forward(hidden_state), baseline)

    def icl(self, *args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if not values['non_streaming_mode'] or state['span'] is not None:
            raise ValueError('Only non-nested non-streaming ICL is supported')
        ref, target = values['ref_id'], values['text_id']
        if ref.shape[1] == 0 or target.shape[1] == 0:
            raise ValueError('Reference and target transcript required')
        state['span'] = (ref.shape[1], ref.shape[1] + target.shape[1])
        try:
            return original_icl(*args, **kwargs)
        finally:
            state['span'] = None

    def generate(self, *args, **kwargs):
        bound = generate_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        prompt = values.get('voice_clone_prompt')
        if (not values.get('non_streaming_mode') or not prompt
                or not prompt.get('icl_mode') or not all(prompt['icl_mode'])
                or prompt.get('ref_code') is None or values.get('ref_ids') is None):
            raise ValueError('Protected adapter requires full-reference non-streaming ICL')
        return original_generate(*args, **kwargs)

    projection.forward = MethodType(routed, projection)
    model.generate_icl_prompt = MethodType(icl, model)
    model.generate = MethodType(generate, model)
    return model


def source_hashes():
    return dict(ORIGINAL_HASHES(), **{'reference_protected.py': pilot.digest(Path(__file__))})


def load_model(protocol):
    wrapper = ORIGINAL_LOAD(protocol)
    protect_reference(wrapper.model)
    cfg = wrapper.model.config
    wrapper.model.talker.text_projection._protected_control_ids = [
        cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]
    return wrapper


def configure_helpers():
    # Explicit dependency injection into the unchanged historical trainer. Child
    # processes always enter THIS module, preserving the same routing in both
    # teacher-forced training and actual generated-speech evaluation.
    pilot.source_hashes = source_hashes
    pilot.load_model = load_model


def audited_load_adapter(talker, state, audit_path):
    """Verify protection on actual nonzero checkpoint weights before synthesis."""
    import torch
    result = ORIGINAL_LOAD_ADAPTER(talker, state)
    projection = talker.text_projection
    was_training = projection.training
    projection.eval()
    try:
        with torch.no_grad():
            device = talker.get_text_embeddings().weight.device
            token_ids = torch.tensor([projection._protected_control_ids + list(range(1, 33))], device=device)
            x = talker.get_text_embeddings()(token_ids)
            baseline = projection.linear_fc2.base_layer(projection.act_fn(projection.linear_fc1.base_layer(x)))
            outside = projection(x)
            projection._target_route['span'] = (19, x.shape[1])
            inside = projection(x)
            control_delta = float((outside - baseline).abs().max())
            reference_delta = float((inside[:, :19] - baseline[:, :19]).abs().max())
            target_rms = float((inside[:, 19:] - baseline[:, 19:]).square().mean().sqrt())
            if control_delta != 0 or reference_delta != 0 or not torch.isfinite(inside).all():
                raise ValueError('Reference/control projection protection failed')
            pilot.write(audit_path, dict(reference_projection_max_abs_change=reference_delta,
                control_projection_max_abs_change=control_delta, target_projection_rms_change=target_rms,
                probe='Three actual TTS control IDs plus fixed text token IDs 1..32; FP32 eval mode',
                interpretation='Direct input projections only; downstream hidden states and perceived voice can still change'))
    finally:
        projection._target_route['span'] = None
        projection.train(was_training)
    return result


def render(run, reports, status):
    pilot.render(run, reports, status)
    path = run/'report.html'
    page = path.read_text(encoding='utf-8').replace('<h1>Targeted retraining</h1>',
        '<h1>Reference-protected pronunciation training</h1>')
    page = page.replace('only text-projection LoRA is trainable.',
        'only target-text LoRA is trainable. Reference transcript and special-token projections bypass the adapter.')
    path.write_text(page, encoding='utf-8')


def orchestrate(cfg, resume=False):
    run = pilot.external(Path(cfg['output']))
    protocol = pilot.read(run/'protocol.json') if resume else pilot.setup(cfg)
    if not resume:
        protocol['criterion'] = ('Lower pooled development WER; no pooled CER increase or seedwise WER increase '
            'in either group; no mean speaker cosine decrease >1e-7 in either group; '
            'no individual cosine decrease >0.005. Listening required before promotion.')
        protocol['approach'] = 'Target-only text projection LoRA; frozen reference transcript, roles and TTS special tokens'
        protocol['comparison'] = 'Same data order, optimizer, checkpoints, prompts, seeds and gates as targeted-voice-pilot'
        pilot.write(run/'protocol.json', protocol)
    if protocol['config'] != cfg or protocol['source'] != source_hashes():
        raise ValueError('Run configuration/source changed')
    reports, eligible, voice_bad, phase = [], [], 0, 'initializing'
    try:
        for step in [0] + cfg['checkpoints']:
            for stage in ('train', 'speech'):
                done = (pilot.checkpoint_path(run, step).exists() if stage == 'train'
                        else (run/f'eval-{step:06d}/results.json').exists())
                if done:
                    continue
                phase = f'{stage}-{step:06d}'
                pilot.write(run/'status.json', dict(status='running', phase=phase, updated=time.time()))
                with (run/f'{phase}.log').open('a', encoding='utf-8') as log:
                    process = subprocess.run([sys.executable, '-m', 'training.reference_protected',
                        '--config', cfg['config_path'], '--stage', stage, '--step', str(step)],
                        cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f'{phase} failed; see its log')
            report = pilot.read(run/f'eval-{step:06d}/results.json')
            reports.append(report)
            if step:
                voice_bad = 0 if report['gate']['voice_pass'] else voice_bad + 1
                if report['gate']['eligible']:
                    eligible.append(report)
            selected = min(eligible, key=lambda r: (r['summary']['development']['pooled_wer'],
                r['summary']['development']['pooled_cer'])) if eligible else reports[0]
            pilot.write(run/'selection.json', dict(selected_step=selected['step'],
                status='candidate_needs_listening' if eligible else 'retain_halfway_baseline',
                evaluated_steps=[r['step'] for r in reports], voice_regression_streak=voice_bad,
                global_model_selection_changed=False))
            render(run, reports, 'running')
            print({'completed_step': step, 'summary': report['summary'], 'gate': report.get('gate')}, flush=True)
            if voice_bad >= cfg['voice_patience']:
                break
        stop = 'voice_guard' if voice_bad >= cfg['voice_patience'] else 'pilot_budget'
        pilot.write(run/'status.json', dict(status='complete', phase='complete', stop_reason=stop,
            evaluated_steps=[r['step'] for r in reports], updated=time.time()))
        render(run, reports, 'complete: ' + stop)
    except BaseException as error:
        pilot.write(run/'status.json', dict(status='failed', phase=phase, error=repr(error), updated=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('run', 'train', 'speech'), default='run')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = pilot.read(args.config)
    cfg['config_path'] = str(args.config.resolve())
    if cfg['checkpoints'] != list(range(cfg['eval_every'], cfg['updates'] + 1, cfg['eval_every'])):
        raise ValueError('Checkpoint schedule mismatch')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NUMBA_CACHE_DIR'] = cfg['numba_cache']
    os.environ['TEMP'] = os.environ['TMP'] = cfg['tmp']
    import torch
    torch.set_num_threads(4)
    configure_helpers()
    if args.stage == 'run':
        orchestrate(cfg, args.resume)
    else:
        protocol = pilot.read(Path(cfg['output'])/'protocol.json')
        if protocol['config'] != cfg:
            raise ValueError('Configuration differs from registered run')
        pilot.load_adapter = lambda talker, state: audited_load_adapter(talker, state,
            Path(cfg['output'])/f'routing-{args.stage}-{args.step:06d}.json')
        (pilot.train if args.stage == 'train' else pilot.speech)(protocol, args.step)


if __name__ == '__main__':
    main()
