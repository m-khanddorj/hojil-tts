"""Conditional fixed-history gradient diagnostic; never fits or selects a policy."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

PROTOCOL_SHA = '89463810f5f9fc5d75d3eabf16144e30872aa1488b0605418f4f604abee12651'
OUTLIER_SHA = 'b9006023fcf1d1584d22f5deb378a1544b744f56c5debd864cc4e7a49f4cbe61'
HELPER_SHA = '612aff5b6088cad3ccd6da7f6ab000cbf458c851efa26a840c8fc6042c051191'
MODES = ('full', 'half')
COMPONENTS = ('main_ce', 'residual_ce', 'main_kl', 'residual_kl')
FLAGS = dict(diagnostic_only=True, optimizer_updates=0, goal_achieved=False,
             global_model_selection_changed=False, data_selection_changed=False)
SETTINGS = dict(teacher_step=0, student_step=32, residual_weight=.3,
    kl_direction='detached_teacher_to_student', kl_arithmetic='float64 log_softmax/probability/sum from native float32 logits',
    teacher_forward='grad-enabled matched semantics; detach outputs and release graph before restoring student; no teacher backward',
    gradient_storage='cpu_float32', geometry_arithmetic='cpu_float64',
    zero_kl_absolute_tolerance=1e-10, zero_gradient_norm_tolerance=1e-7,
    component_gradient_atol=1e-6, component_gradient_rtol=2e-4,
    modes=list(MODES), pair_weight='equal within each reported stratum', mode_weight=.5)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def helper(path):
    require(digest(path) == HELPER_SHA, 'Parent evidence helper changed')
    spec = importlib.util.spec_from_file_location('gradient_probe_parent_evidence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failed_terminal(status, closure, history):
    """Pure additional gate; helper.parent_evidence independently binds artifacts."""
    from .reference_masked_supervision_evaluation import progression
    require(status.get('status') == 'complete' and status.get('step') == 32,
            'Completed terminal32 required')
    require(closure.get('status') == 'closed' and type(closure.get('exit_code')) is int
        and closure['exit_code'] == 0 and type(closure.get('original_session_id')) is int
        and closure['original_session_id'] == 33720 and closure.get('step') == 32,
        'Authoritative original33720 closed0 receipt required')
    require([r['step'] for r in history] == [0, 8, 16, 32], 'Complete80 history through32 required')
    decision = progression(history)
    require(decision['stop'] and decision['step'] == 32 and decision['reason'] in
        ('maximum_updates_reached', 'consecutive_pareto_regressions')
        and status.get('progression') == decision and status.get('reason') == decision['reason'],
        'Failed final checkpoint required; qualification must go to original114')
    require(status.get('goal_achieved') is False and status.get('global_model_selection_changed') is False,
            'Unexpected parent promotion')
    return decision


def panel(schedule, fitting, outliers):
    fixed = schedule['fitting_probe_pair_ids']
    flags = outliers['outlier_pairs']
    indexed = {r['pair_id']: r for r in flags}
    require(len(fixed) == len(set(fixed)) == 16 and len(flags) == len(indexed) == 15,
            'Exact fixed16 and existing flagged15 required')
    union = sorted(set(fixed) | set(indexed))
    require(len(union) == 30 and len(set(fixed) & set(indexed)) == 1, 'Exact deduplicated30 panel required')
    result = []
    for identity in union:
        require(identity in fitting, 'Unknown fitting pair')
        row = fitting[identity]
        require(row['id'] == identity and row['partition'] == 'fitting' and row['ref_id'] != identity,
                'Evaluation/calibration/self-reference inputs prohibited')
        flag = indexed.get(identity)
        if flag is not None:
            require(flag['reference_id'] == row['ref_id'] and flag['speaker_id'] == row['speaker_id']
                and math.isfinite(flag['positive_minus_strongest_negative'])
                and flag['positive_minus_strongest_negative'] < 0, 'Existing acoustic flag metadata differs')
        result.append(dict(row=row, fixed_probe=identity in fixed, flagged=flag is not None,
                           acoustic_flag=flag))
    return result


def parent_evidence(train, helper_path):
    train = Path(train)
    require(digest(train/'protocol.json') == PROTOCOL_SHA, 'Wrong registered parent')
    p = read(train/'protocol.json')
    api = helper(helper_path)
    require(train.resolve() == api.TRAIN.resolve(), 'Parent helper scope differs')
    evidence, summary = api.parent_evidence(p)
    history = [read(train/f'verification-{step:06d}.json') for step in (0, 8, 16, 32)]
    failed_terminal(read(api.PARENT_STATUS), read(api.CLOSURE), history)
    require(summary['step'] == 32, 'Parent stopped before32')
    return p, evidence, summary


def prepare(train, out, driver, helper_path, test_log):
    train, out = Path(train), Path(out)
    require(out.parent.resolve() == train.parent.resolve() and out.name == 'masked-supervision-gradient-probe-v1'
        and not out.exists(), 'New external diagnostic directory required; no restart/overwrite')
    p, evidence, parent = parent_evidence(train, helper_path)
    schedule = read(p['plan'])
    outlier_path = train.parent/'masked-fitting-voice-outliers-v1.json'
    require(digest(outlier_path) == OUTLIER_SHA, 'Predetermined fitting flags changed')
    outliers = read(outlier_path)
    require(outliers['training_protocol_sha256'] == PROTOCOL_SHA
        and outliers['plan_sha256'] == p['plan_sha256']
        and outliers['feature_protocol_sha256'] == p['feature_protocol_sha256']
        and outliers['feature_audit_sha256'] == p['feature_audit_sha256'], 'Flag provenance differs')
    cases = panel(schedule, p['fitting_rows'], outliers)
    feature_run = Path(p['features']); feature_protocol = read(feature_run/'protocol.json')
    feature_audit = read(feature_run/'verification.json'); extraction = read(feature_run/'extraction.json')
    assets = {r['id']: r for r in feature_protocol['records']}
    ids = {r['row'][key] for r in cases for key in ('id', 'ref_id')}
    for identity in sorted(ids):
        directory = feature_run/'assets'/hashlib.sha256(identity.encode()).hexdigest()
        for path, sha in ((directory/'features.pt', feature_audit['feature_sha256'][identity]),
                          (directory/'receipt.json', extraction['asset_receipts'][identity])):
            require(digest(path) == sha, 'Registered native feature changed')
            evidence[str(path.resolve())] = sha
        require(identity in assets, 'Missing audited feature descriptor')
    paths = [Path(__file__), Path(__file__).parents[1]/'tests/test_reference_masked_supervision_gradient_probe.py',
        Path(driver), Path(helper_path), Path(test_log), outlier_path, feature_run/'extraction.json']
    log = Path(test_log).read_text(encoding='utf-8-sig')
    require(any(line.startswith('Ran 9 tests in ') for line in log.splitlines())
        and log.strip().endswith('OK') and 'FAILED' not in log, 'Nine passing focused tests required')
    for path in paths:
        evidence[str(path.resolve())] = digest(path)
    protocol = dict(kind='fixed_real_history_component_gradient_diagnostic', train=str(train.resolve()),
        helper=str(Path(helper_path).resolve()), parent=parent, evidence=evidence, cases=cases,
        settings=SETTINGS, training_protocol_sha256=PROTOCOL_SHA, features=p['features'],
        feature_protocol_sha256=p['feature_protocol_sha256'], feature_audit_sha256=p['feature_audit_sha256'],
        training_hardware=p['training_hardware'], numeric_config=p['numeric_config'],
        runtime=p['runtime'], parameter_scope='NEW masked Q/V A/B only', **FLAGS)
    out.mkdir(); write_new(out/'protocol.json', protocol)
    return protocol


def inherited_fields(p, parent):
    require(p.get('training_protocol_sha256') == PROTOCOL_SHA and all(p.get(k) == parent[k]
        for k in ('features', 'feature_protocol_sha256', 'feature_audit_sha256',
                  'training_hardware', 'numeric_config', 'runtime')), 'Inherited native feature/runtime identity changed')


def require_protocol_hash(out, expected):
    require(digest(Path(out)/'protocol.json') == expected, 'Registered diagnostic protocol changed')


def verify(out):
    out = Path(out); p = read(out/'protocol.json')
    require(p['settings'] == SETTINGS and all(p.get(k) == v for k, v in FLAGS.items()), 'Diagnostic recipe changed')
    for path, sha in p['evidence'].items():
        require(digest(path) == sha, 'Changed diagnostic evidence: '+path)
    parent, evidence, receipt = parent_evidence(Path(p['train']), p['helper'])
    inherited_fields(p, parent)
    require(receipt == p['parent'] and all(p['evidence'].get(k) == v for k, v in evidence.items()), 'Parent evidence changed')
    require(p['cases'] == panel(read(parent['plan']), parent['fitting_rows'],
        read(Path(p['train']).parent/'masked-fitting-voice-outliers-v1.json')), 'Fixed fitting panel changed')
    return p, parent


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


@contextmanager
def preserved(core):
    """Restore loaded student adapters, original gradients and all RNG on failure too."""
    from . import reference_masked_supervision_policy_v2 as policy
    from . import reference_masked_supervision_training_v2 as fit
    state = policy.snapshot(core); rng = fit.rng_state()
    parameters = core.parameters(); grads = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    try:
        yield
    finally:
        policy.restore(core, state)
        for parameter, gradient in zip(parameters, grads):
            parameter.grad = gradient
        fit.restore_rng(rng)
        require(equal(policy.snapshot(core), state) and equal(fit.rng_state(), rng), 'State restoration failed')


def forward_kl(student, teacher):
    import torch
    require(student.dtype == teacher.dtype == torch.float32 and student.shape == teacher.shape
        and student.ndim in (2, 3) and student.shape[-1] > 1
        and torch.isfinite(student).all() and torch.isfinite(teacher).all(), 'Invalid full-vocabulary logits')
    t = teacher.detach().to(student.device).double().log_softmax(-1)
    s = student.double().log_softmax(-1)
    value = (t.exp() * (t-s)).sum(-1).mean()
    require(torch.isfinite(value) and float(value.detach()) >= -1e-10, 'Nonfinite or negative KL')
    return value


def alignment(core, seq, output, item):
    import torch
    from .reference_masked_supervision_policy_v2 import tensor_digest
    frames = len(item['codes']); main, residual = output['main_logits'], output['residual_logits']
    labels = seq.labels[seq.labels != -100]
    expected = torch.cat([seq.codes[:, 0], labels.new_tensor([core.model.config.talker_config.codec_eos_token_id])])
    require(main.dtype == residual.dtype == torch.float32 and main.ndim == 2
        and main.shape[0] == frames+1 and residual.ndim == 3 and residual.shape[:2] == (frames, 15)
        and output['token_count'] == 16*frames+1 and torch.equal(labels, expected)
        and torch.equal(seq.codes.cpu(), item['codes'].cpu()), 'Native all16/EOS preframe alignment changed')
    tensors = [('embeddings', seq.embeddings), ('labels', seq.labels), ('codes', seq.codes),
        ('input_ids', item['input_ids']), ('reference_ids', item['reference'].ref_ids),
        ('reference_codes', item['reference'].ref_codes), ('original_reference_xvec', item['reference'].speaker)]
    return dict(prefix_length=seq.prefix_length, frames=frames, main_tokens=frames+1,
        residual_tokens=15*frames, main_vocabulary=main.shape[-1], residual_vocabulary=residual.shape[-1],
        native_input_sha256=tensor_digest(tensors), reference_sha256=item['reference'].reference_sha256)


def teacher_outputs(core, item, mode, teacher, student):
    import torch
    from . import reference_masked_supervision_policy_v2 as policy
    require(core.scope is None and mode in MODES and item['partition'] == 'fitting', 'Fitting-only inactive scope required')
    # No student graph may exist before these in-place state restorations.
    policy.restore(core, teacher)
    try:
        # Match student forward/checkpoint/attention semantics; no_grad can pick
        # different kernels. No teacher backward: detach and release this graph
        # completely before the in-place student restoration in finally.
        with torch.enable_grad(), core.replay(item['input_ids'], item['codes'], item['reference'], mode == 'half', terminated=True) as seq:
            output = core.outputs(seq); trace = alignment(core, seq, output, item)
            logits = {k: output[k].detach().cpu().clone() for k in ('main_logits', 'residual_logits')}
            del output
        del seq
    finally:
        policy.restore(core, student)
    return logits, trace


def component_gradient(core, item, mode, component, teacher=None, expected_trace=None):
    import torch
    require(item['partition'] == 'fitting' and mode in MODES and component in (*COMPONENTS, 'task'),
            'Fitting-only registered component required')
    parameters = core.parameters()
    for parameter in parameters:
        parameter.grad = None
    try:
        with core.replay(item['input_ids'], item['codes'], item['reference'], mode == 'half', terminated=True) as seq:
            output = core.outputs(seq); trace = alignment(core, seq, output, item)
            require(expected_trace is None or trace == expected_trace, 'Teacher/student native conditioning differs')
            frames = len(item['codes'])
            if component == 'main_ce':
                loss = output['main_nll']/(frames+1)
            elif component == 'residual_ce':
                loss = output['residual_nll']/(15*frames)
            elif component == 'task':
                loss = output['main_nll']/(frames+1)+.3*output['residual_nll']/(15*frames)
            else:
                key = 'main_logits' if component == 'main_kl' else 'residual_logits'
                require(teacher is not None, 'Detached fixed teacher logits required')
                loss = forward_kl(output[key], teacher[key])
            require(torch.isfinite(loss), 'Nonfinite diagnostic loss')
            value = float(loss.detach()); loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), 'Incomplete/nonfinite NEW gradient')
            gradient = torch.cat([p.grad.detach().cpu().flatten() for p in parameters])
            del loss, output
        del seq
        return value, gradient, trace
    finally:
        for parameter in parameters:
            parameter.grad = None


def geometry(values):
    import torch
    require(set(values) == set(COMPONENTS), 'Component coverage differs')
    vectors = {k: v.detach().cpu().double() for k, v in values.items()}
    require(len({v.shape for v in vectors.values()}) == 1 and all(torch.isfinite(v).all() for v in vectors.values()), 'Invalid gradient geometry')
    vectors['weighted_residual_ce'] = .3*vectors['residual_ce']
    vectors['task'] = vectors['main_ce']+vectors['weighted_residual_ce']
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    pairs = {}
    for i, a in enumerate(vectors):
        for b in list(vectors)[i+1:]:
            dot = float(torch.dot(vectors[a], vectors[b])); denom = norms[a]*norms[b]
            pairs[a+'__'+b] = dict(dot=dot, cosine=dot/denom if denom else None,
                                  opposing=dot < 0, negative_gradient_directional_dot=-dot)
    return dict(norms=norms, comparisons=pairs,
        weighted_residual_to_main_norm=norms['weighted_residual_ce']/norms['main_ce'] if norms['main_ce'] else None)


def controls(core, item, teacher, student):
    import torch
    from . import reference_masked_supervision_policy_v2 as policy
    rows = []
    with preserved(core):
        for mode in MODES:
            logits, trace = teacher_outputs(core, item, mode, teacher, teacher)
            zero = {}
            for name in ('main_kl', 'residual_kl'):
                loss, grad, _ = component_gradient(core, item, mode, name, logits, trace)
                norm = float(grad.double().norm())
                require(abs(loss) <= SETTINGS['zero_kl_absolute_tolerance']
                    and norm <= SETTINGS['zero_gradient_norm_tolerance'], 'Identical-policy KL/gradient control failed')
                zero[name] = dict(value=loss, gradient_norm=norm)
            policy.restore(core, student)
            _, main, _ = component_gradient(core, item, mode, 'main_ce', expected_trace=trace)
            _, residual, _ = component_gradient(core, item, mode, 'residual_ce', expected_trace=trace)
            _, combined, _ = component_gradient(core, item, mode, 'task', expected_trace=trace)
            expected = main+.3*residual
            require(torch.allclose(combined, expected, atol=SETTINGS['component_gradient_atol'],
                rtol=SETTINGS['component_gradient_rtol']), 'Combined task gradient differs from components')
            rows.append(dict(mode=mode, zero=zero, combined_maximum_error=float((combined-expected).abs().max()), trace=trace))
    return rows


def aggregate(records, panel_rows):
    import torch
    require(len(records) == 2*len(panel_rows) and len({(r['id'], r['mode']) for r in records}) == len(records),
            'Complete unique pair/mode gradient coverage required')
    by_id = {r['row']['id']: r for r in panel_rows}
    require({(r['id'], r['mode']) for r in records} == {(i, m) for i in by_id for m in MODES}, 'Gradient panel differs')
    result = {}
    for stratum in ('all', 'fixed_probe', 'flagged', 'unflagged'):
        ids = {i for i, r in by_id.items() if stratum == 'all' or
               (not r['flagged'] if stratum == 'unflagged' else r[stratum])}
        require(ids, 'Empty diagnostic stratum')
        result[stratum] = {}
        for mode in (*MODES, 'both'):
            selected = [r for r in records if r['id'] in ids and (mode == 'both' or r['mode'] == mode)]
            vectors = {k: sum((r['gradients'][k].double() for r in selected), torch.zeros_like(selected[0]['gradients'][k], dtype=torch.float64))/len(selected) for k in COMPONENTS}
            per_pair = [geometry(r['gradients']) for r in selected]
            result[stratum][mode] = dict(pairs=len(ids), modes=2 if mode == 'both' else 1,
                weight=1/len(selected), geometry=geometry(vectors),
                mean_component_values={k: sum(r['losses'][k] for r in selected)/len(selected) for k in COMPONENTS},
                mean_individual_norms={k: sum(g['norms'][k] for g in per_pair)/len(selected) for k in COMPONENTS},
                observation_count=len(selected), observation_unit='pair_mode',
                opposing_pair_mode_counts={k: sum(g['comparisons'][k]['opposing'] for g in per_pair) for k in per_pair[0]['comparisons']})
    return result


def run(out, protocol_sha):
    import torch
    from . import reference_masked_supervision_training_v2 as fit
    from . import reference_masked_supervision_policy_v2 as policy
    from . import reference_masked_supervision_features as features
    out = Path(out); require_protocol_hash(out, protocol_sha); p, parent = verify(out)
    require(not (out/'started.json').exists(), 'No implicit diagnostic restart')
    fit.configure(); require(fit.hardware() == p['training_hardware'], 'Registered single-GPU hardware required')
    write_new(out/'started.json', dict(protocol_sha256=digest(out/'protocol.json'), **FLAGS))
    records = []; restored = False
    try:
        _, wrapper, core, student_state = fit.load(Path(p['train']), 32, 'cuda', training=True)
        teacher_state = fit.read_checkpoint(Path(p['train']), 0)
        _, assets = features.load_features(p['features'])
        require(policy.frozen_identity(wrapper.model, core) == parent['frozen_identity'], 'Frozen policy mismatch')
        student = policy.snapshot(core); teacher = teacher_state['adapter']
        require(equal(student, student_state['adapter']) and student_state['step'] == 32 and teacher_state['step'] == 0,
                'Wrong student/teacher snapshots')
        before_rng = fit.rng_state(); frozen = policy.frozen_identity(wrapper.model, core)
        with preserved(core):
            first = fit.pair(next(r['row'] for r in p['cases'] if r['fixed_probe']), assets)
            checks = controls(core, first, teacher, student)
            write_new(out/'controls.json', dict(probe_id=first['id'], controls=checks, **FLAGS))
            for index, registered in enumerate(p['cases']):
                item = fit.pair(registered['row'], assets)
                for mode in MODES:
                    print(f'Gradient diagnostic {index+1}/30 {item["id"]} {mode}', flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    logits, trace = teacher_outputs(core, item, mode, teacher, student)
                    values, grads = {}, {}
                    for component in COMPONENTS:
                        values[component], grads[component], _ = component_gradient(core, item, mode, component, logits, trace)
                    path = out/f'gradients-{index:03d}-{mode}.pt'
                    torch.save(dict(gradients=grads, parameter_names=list(core.adapters.state_dict()), **FLAGS), path)
                    row = dict(id=item['id'], mode=mode, losses=values, geometry=geometry(grads), trace=trace,
                        fixed_probe=registered['fixed_probe'], flagged=registered['flagged'], acoustic_flag=registered['acoustic_flag'],
                        gradients_path=path.name, gradients_sha256=digest(path), peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    write_new(out/f'pair-{index:03d}-{mode}.json', row)
                    records.append(dict(row, gradients=grads)); del logits
            summaries = aggregate(records, p['cases'])
        restored = True
        require(equal(policy.snapshot(core), student) and equal(fit.rng_state(), before_rng)
            and policy.frozen_identity(wrapper.model, core) == frozen and core.scope is None
            and all(v.grad is None for v in core.parameters()), 'Diagnostic changed student/frozen/RNG/gradient state')
        require_protocol_hash(out, protocol_sha); verify(out)
        report = dict(status='verified', protocol_sha256=digest(out/'protocol.json'),
            controls_sha256=digest(out/'controls.json'), pairs=30, pair_mode_records=60,
            records=[{k:v for k,v in r.items() if k != 'gradients'} for r in records], aggregates=summaries,
            parameter_names=list(core.adapters.state_dict()), parameter_count=sum(v.numel() for v in core.parameters()),
            teacher_adapter_sha256=policy.tensor_digest(teacher.items()), student_adapter_sha256=policy.tensor_digest(student.items()),
            frozen_identity=frozen, adapters_restored=True, rng_restored=True, gradients_restored=True,
            original_checkpoints_and_optimizer_files_unchanged=True,
            limitation='Enriched fitting-only fixed-history mechanism diagnostic; no speech-quality or corpus-prevalence claim', **FLAGS)
        write_new(out/'results.json', report)
        return report
    except BaseException as error:
        write_new(out/'failure.json', dict(error=type(error).__name__, message=str(error),
            complete_state_restoration_verified=restored, automatic_retry=False, **FLAGS))
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--driver', type=Path)
    parser.add_argument('--helper', type=Path)
    parser.add_argument('--test-log', type=Path)
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.train, args.out, args.driver, args.helper, args.test_log)
    else:
        run(args.out, args.protocol_sha)


if __name__ == '__main__':
    main()
