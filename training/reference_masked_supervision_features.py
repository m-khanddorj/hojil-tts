"""Fresh native, unadapted CPU FP32 features for the fixed masked-SFT pilot.

Registration is model independent. Extraction is an explicit separate command;
never reads historical tensor caches or chooses a talker policy. All artifacts
must live outside the repository. A completed native audit is required by readers.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import wave

PLAN_SHA = 'a4cefa5a77ca97696048df57a31f023c5a0441f88a626811e46edd13470c0c34'
DATA_SHA = '89b0819de650369fb03556fd190a4b7129e58bfc32ebdfd5886e09addb98e62d'
AUDIT_SHA = '284091de8f2a133e09fbb32c920dd4ef5e9ed068dc8fa79df67e134efea38057'
CUSTOM_SHA = '860372aef1859fa8295d68a16a20fa0b0f9bad2e19b30048da0ee1039b32827d'
CUSTOM_ID = '__custom_reference__'
REPO = Path(__file__).resolve().parents[1]
SETTINGS = dict(device='cpu', dtype='float32', attention='sdpa', cpu_threads=4,
                matmul_tf32=False, cudnn_tf32=False, seed=20261111,
                real_assets=323, total_assets=324, codebooks=16, speaker_size=1024)
TENSORS = ('input_ids', 'ref_ids', 'codes', 'speaker')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), f'Refusing overwrite: {path}')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def bound_files(mapping):
    for path, expected in mapping.items():
        require(digest(path) == expected, f'Bound file changed: {path}')


def external(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(REPO), 'Feature artifacts must stay outside Git')
    return path


def wav_header(path):
    with wave.open(str(path), 'rb') as stream:
        result = dict(sample_rate=stream.getframerate(), channels=stream.getnchannels(),
                      samples=stream.getnframes(), sample_width=stream.getsampwidth())
        require(stream.getcomptype() == 'NONE', 'Uncompressed real WAV required')
    require(result['sample_rate'] == 24000 and result['channels'] == 1
            and result['samples'] > 0, 'Expected complete mono 24kHz recording')
    return result


def descriptors(plan, fitting, calibration, custom):
    """Dedupe by utterance ID and reject conflicting repeated reference metadata."""
    index = {r['id']: r for r in fitting}
    require(len(index) == len(fitting), 'Duplicate fitting ID')
    require(set(plan['selected_fitting_pair_ids']) <= set(index), 'Unplanned fitting ID')
    consumed = [index[i] for i in plan['selected_fitting_pair_ids']] + calibration
    assets = {}
    for row in consumed:
        for prefix in ('', 'ref_'):
            item = dict(id=row[prefix+'id'], text=row[prefix+'text'],
                audio=str(Path(row[prefix+'audio']).resolve()),
                audio_sha256=row[prefix+'audio_sha256'], speaker_id=row['speaker_id'],
                sample_rate=row['sample_rate'], source=row['source'],
                kind='real_speech')
            require(item['text'].strip() and item['sample_rate'] == 24000, 'Invalid real descriptor')
            if item['id'] in assets:
                require(assets[item['id']] == item, 'Conflicting same-utterance descriptor')
            assets[item['id']] = item
    require(set(assets) == set(plan['real_feature_ids']), 'Feature union differs from fixed plan')
    require(custom['reference_sha256'] == CUSTOM_SHA, 'Wrong custom reference')
    require(CUSTOM_ID not in assets and CUSTOM_SHA not in {a['audio_sha256'] for a in assets.values()},
            'Custom reference must be distinct from real supervised targets')
    assets[CUSTOM_ID] = dict(id=CUSTOM_ID, text=custom['reference_text'],
        audio=str(Path(custom['reference']).resolve()), audio_sha256=CUSTOM_SHA,
        speaker_id=custom['speaker_id'], sample_rate=24000, source='user_reference',
        kind='reference_only_no_real_target')
    # An ID alias must not silently duplicate a real audio extraction or transcript.
    require(len({a['audio_sha256'] for a in assets.values()}) == len(assets),
            'Distinct IDs alias the same audio bytes')
    return [assets[i] for i in sorted(assets)]


def resolve_inputs(plan_path):
    plan_path = Path(plan_path).resolve()
    require(digest(plan_path) == PLAN_SHA, 'Unregistered fixed data plan')
    plan = read(plan_path)
    data_path = Path(plan['data_protocol']); data = data_path.parent
    require(digest(data_path) == DATA_SHA == plan['data_protocol_sha256'], 'Data protocol changed')
    require(digest(data/'audit.json') == AUDIT_SHA == plan['data_audit_sha256'], 'Data audit changed')
    p = read(data_path); audit = read(data/'audit.json')
    require(audit['protocol_sha256'] == DATA_SHA and audit['status'] == 'verified_manifest_and_audio_hashes_only',
            'Wrong data audit')
    evidence = {str(plan_path):PLAN_SHA, str(data_path):DATA_SHA, str(data/'audit.json'):AUDIT_SHA}
    for name, sha in plan['manifest_sha256'].items():
        require(p['manifests'][name] == sha == audit['manifest_sha256'][name], 'Manifest binding differs')
        evidence[str(data/name)] = sha
    evidence.update(p['inputs']); evidence.update(p['source_sha256']); evidence.update(plan['sources'])
    bound_files(evidence)
    initial_path = next(Path(x) for x in p['inputs'] if Path(x).parent.name == 'middle-attention-pilot')
    panel_path = next(Path(x) for x in p['inputs'] if Path(x).parent.name == 'joint-goal-confirmation')
    initial = read(initial_path); panel = read(panel_path)
    require(initial['pool_protocol_sha256'] == digest(Path(initial['pool'])/'protocol.json'), 'Speaker split changed')
    custom = [c for c in panel['cases'] if c['group'] == 'custom']
    fields = ('reference', 'reference_sha256', 'reference_text', 'speaker_id')
    require(len(custom) == 13 and all(all(c[k] == custom[0][k] for k in fields) for c in custom),
            'Custom original-panel conditioning differs')
    from .reference_masked_supervision_plan import build_plan
    fitting, calibration = rows(data/'train.jsonl'), rows(data/'calibration.jsonl')
    require(all(plan[k] == v for k,v in build_plan(fitting, calibration).items()), 'Fixed schedule differs')
    records = descriptors(plan, fitting, calibration, custom[0])
    require(len(records) == 324 and len(plan['real_feature_ids']) == 323, 'Expected exact 324 assets')
    for item in records:
        require(digest(item['audio']) == item['audio_sha256'], 'Real/custom WAV changed')
        item['wav_header'] = wav_header(item['audio'])
    return records, evidence, initial['config']


def sources(driver, test_log):
    files = [Path(__file__), REPO/'training/reference_masked_supervision_plan.py',
             REPO/'tests/test_reference_masked_supervision_features.py', Path(driver), Path(test_log)]
    return {str(p.resolve()):digest(p) for p in files}


def runtime():
    packages = ('qwen-tts','torch','transformers','torchaudio','numpy','soundfile',
                'librosa','safetensors','tokenizers','accelerate')
    versions = {name:importlib.metadata.version(name) for name in packages}
    qwen = Path(importlib.metadata.distribution('qwen-tts').locate_file('qwen_tts'))
    require(qwen.is_dir(), 'Native Qwen implementation missing')
    code = {str(p.resolve()):digest(p) for p in sorted(qwen.rglob('*.py'))}
    records = {}
    for name in packages:
        distribution = importlib.metadata.distribution(name)
        record = next((x for x in distribution.files or [] if str(x).endswith('.dist-info/RECORD')), None)
        if record is not None:
            path = distribution.locate_file(record); records[str(path.resolve())] = digest(path)
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                platform=platform.platform(), versions=versions, native_source_sha256=code,
                package_record_sha256=records)


def file_tree(path):
    return {str(p.resolve()):digest(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}


def safetensor_non_talker(path):
    """Byte-check stored non-talker components without importing or loading tensors."""
    result = {}
    with Path(path).open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        require(0 < size < 32 * 2**20, 'Unexpected safetensors header')
        header = json.loads(stream.read(size)); offset = 8 + size
        for name, item in sorted(header.items()):
            if name == '__metadata__' or name.startswith('talker.'):
                continue
            start, stop = item['data_offsets']; require(0 <= start < stop, 'Bad tensor range')
            stream.seek(offset + start); remain = stop-start; sha = hashlib.sha256()
            while remain:
                chunk = stream.read(min(remain, 2**20)); require(chunk, 'Truncated tensor storage')
                sha.update(chunk); remain -= len(chunk)
            result[name] = dict(dtype=item['dtype'], shape=item['shape'], sha256=sha.hexdigest())
    require(result and all(n.startswith('speaker_encoder.') for n in result), 'Unknown non-talker state')
    return result


def prepare(plan, run, driver, test_log):
    run = external(run); require(not run.exists(), 'Do not overwrite feature registration')
    records, evidence, config = resolve_inputs(plan)
    log = Path(test_log).read_text(encoding='utf-8-sig')
    require('Ran 5 tests' in log and '\nOK\n' in log and 'skipped' not in log.lower(), 'Feature tests missing')
    base, adapted = Path(config['base']), Path(config['adapted'])
    require(base.name == '5d83992436eae1d760afd27aff78a71d676296fc', 'Wrong native base revision')
    models = file_tree(base)
    speaker = safetensor_non_talker(base/'model.safetensors')
    require(speaker == safetensor_non_talker(adapted/'model.safetensors'), 'Historical SFT changed raw speaker encoder')
    adapted_files = {str((adapted/'model.safetensors').resolve()):digest(adapted/'model.safetensors')}
    for path in sorted((base/'speech_tokenizer').rglob('*')):
        if path.is_file():
            other = adapted/'speech_tokenizer'/path.relative_to(base/'speech_tokenizer')
            require(digest(other) == digest(path), 'Historical policy changed codec')
            adapted_files[str(other.resolve())] = digest(other)
    codec = read(base/'speech_tokenizer/config.json'); model = read(base/'config.json')
    require(codec['encoder_valid_num_quantizers'] == 16 and codec['encoder_config']['codebook_size'] == 2048
            and model['speaker_encoder_config']['enc_dim'] == 1024, 'Native tensor contract differs')
    protocol = dict(kind='fresh_native_reference_features', settings=SETTINGS, plan=str(Path(plan).resolve()),
        records=records, evidence=evidence, sources=sources(driver,test_log), runtime=runtime(),
        base=str(base.resolve()), model_files=models, historical_non_talker_files=adapted_files,
        historical_speaker_state=speaker, codebook_size=2048,
        driver=str(Path(driver).resolve()), test_log=str(Path(test_log).resolve()),
        legacy_cache_used=False, talker_policy_selected=False, training_updates=0,
        native_tokenizer='Original assistant and reference framing through native processor',
        preprocessing='Full untrimmed pinned WAV through native create_voice_clone_prompt; no VAD, clipping, or target truncation',
        speaker_policy='Unadapted original FP32 vector; downstream policy applies its speaker adapter once')
    run.mkdir(); write_new(run/'protocol.json', protocol)
    return protocol


def verify(run):
    run = external(run); p = read(run/'protocol.json')
    require(p['settings'] == SETTINGS and p['kind'] == 'fresh_native_reference_features', 'Feature settings changed')
    require(p['sources'] == sources(p['driver'],p['test_log']) and p['runtime'] == runtime(), 'Feature execution changed')
    bound_files(p['evidence']); bound_files(p['historical_non_talker_files'])
    require(p['model_files'] == file_tree(p['base']), 'Model/tokenizer file inventory changed')
    records, evidence, config = resolve_inputs(p['plan'])
    require(records == p['records'] and evidence == p['evidence']
            and Path(config['base']).resolve() == Path(p['base']), 'Feature descriptors changed')
    return p


def configure_numeric():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SETTINGS['seed'])
    require(torch.get_num_threads() == 4 and not torch.backends.cuda.matmul.allow_tf32
            and not torch.backends.cudnn.allow_tf32, 'Numeric settings differ')


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    sha = hashlib.sha256(f'{value.dtype}:{list(value.shape)}:'.encode())
    sha.update(value.numpy().tobytes()); return sha.hexdigest()


def native_component_identity(wrapper):
    result = {}
    for label, module in [('speaker_encoder',wrapper.model.speaker_encoder),
                          ('speech_tokenizer',wrapper.model.speech_tokenizer.model)]:
        state = module.state_dict()
        require(state, 'Native component has no state')
        result[label] = {name:tensor_sha256(tensor) for name,tensor in sorted(state.items())}
    return result


def native_ids(processor, text):
    from qwen_tts import Qwen3TTSModel
    result = {}
    for name, method in [('input_ids',Qwen3TTSModel._build_assistant_text),
                         ('ref_ids',Qwen3TTSModel._build_ref_text)]:
        value = processor(text=method(None,text),return_tensors='pt',padding=True)['input_ids'].cpu()
        result[name] = value.unsqueeze(0) if value.ndim == 1 else value
    return result


def load_native_processor(base):
    from qwen_tts.core.models import Qwen3TTSProcessor
    # Match the native wrapper's AutoProcessor loading policy exactly, including
    # the tokenizer regex compatibility fix, without loading model weights.
    return Qwen3TTSProcessor.from_pretrained(str(base),local_files_only=True,fix_mistral_regex=True)


def validate_feature(value, descriptor, protocol_sha, codebook_size=2048):
    import torch
    require(set(value) == set(TENSORS) | {'id','audio_sha256','protocol_sha256','descriptor_sha256'},
            'Unexpected feature payload keys')
    require(value['id'] == descriptor['id'] and value['audio_sha256'] == descriptor['audio_sha256']
            and value['protocol_sha256'] == protocol_sha
            and value['descriptor_sha256'] == fingerprint(descriptor), 'Feature identity differs')
    for name, minimum in [('input_ids',8),('ref_ids',5)]:
        tensor = value[name]
        require(isinstance(tensor,torch.Tensor) and tensor.dtype == torch.long
                and tensor.ndim == 2 and tensor.shape[0] == 1 and tensor.shape[1] > minimum
                and tensor.device.type == 'cpu' and not tensor.requires_grad
                and bool((tensor >= 0).all()), 'Invalid native transcript tensor')
    codes, speaker = value['codes'],value['speaker']
    require(isinstance(codes,torch.Tensor) and codes.dtype == torch.long and codes.device.type == 'cpu'
            and codes.ndim == 2 and codes.shape[1] == 16 and len(codes) > 0
            and bool(((codes >= 0)&(codes < codebook_size)).all()), 'Invalid complete native codec frames')
    require(isinstance(speaker,torch.Tensor) and speaker.dtype == torch.float32
            and speaker.device.type == 'cpu' and speaker.shape == (1024,)
            and not speaker.requires_grad and bool(torch.isfinite(speaker).all())
            and math.isfinite(float(speaker.norm())) and float(speaker.norm()) > 0,
            'Original finite FP32 speaker vector required')
    return {name:dict(shape=list(value[name].shape),dtype=str(value[name].dtype),
                     sha256=tensor_sha256(value[name])) for name in TENSORS}


def asset_key(identity):
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()


def read_asset(run, descriptor, protocol_sha):
    import torch
    directory = Path(run)/'assets'/asset_key(descriptor['id'])
    receipt = read(directory/'receipt.json')
    require(receipt['id'] == descriptor['id'] and receipt['protocol_sha256'] == protocol_sha
            and receipt['descriptor_sha256'] == fingerprint(descriptor), 'Asset receipt identity changed')
    require(digest(directory/'features.pt') == receipt['feature_sha256'], 'Feature file changed')
    value = torch.load(directory/'features.pt',map_location='cpu',weights_only=True)
    require(validate_feature(value,descriptor,protocol_sha) == receipt['tensors'], 'Tensor receipt differs')
    return value,receipt


def extract(run):
    import torch
    from qwen_tts import Qwen3TTSModel
    run = external(run); p = verify(run); protocol_sha = digest(run/'protocol.json')
    require(not (run/'extraction.json').exists() and not (run/'verification.json').exists(), 'Extraction already complete')
    configure_numeric(); assets = run/'assets'; assets.mkdir(exist_ok=True)
    expected = {asset_key(d['id']) for d in p['records']}
    require(all(x.is_dir() and x.name in expected for x in assets.iterdir()), 'Uncommitted or foreign asset; refuse unsafe resume')
    wrapper = Qwen3TTSModel.from_pretrained(p['base'],device_map='cpu',torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True)
    wrapper.model.requires_grad_(False).eval()
    wrapper.model.speech_tokenizer.model.requires_grad_(False).eval()
    components = (wrapper.model,wrapper.model.speech_tokenizer.model)
    require(all(v.device.type == 'cpu' and (not v.is_floating_point() or v.dtype == torch.float32)
        for m in components for v in m.state_dict().values()), 'All native inference must be CPU FP32')
    identity = native_component_identity(wrapper)
    versions = [(v,v._version) for m in components for v in list(m.parameters())+list(m.buffers())]
    receipts = {}
    for number, descriptor in enumerate(p['records'],1):
        key = asset_key(descriptor['id']); directory = assets/key
        if directory.exists():
            value,receipt = read_asset(run,descriptor,protocol_sha)
            require(receipt['native_components'] == fingerprint(identity), 'Resume native component differs')
        else:
            require(digest(descriptor['audio']) == descriptor['audio_sha256'], 'Source WAV changed')
            with torch.inference_mode():
                prompt = wrapper.create_voice_clone_prompt(ref_audio=descriptor['audio'],
                    ref_text=descriptor['text'],x_vector_only_mode=False)
            require(len(prompt) == 1 and prompt[0].icl_mode and not prompt[0].x_vector_only_mode
                    and prompt[0].ref_text == descriptor['text'], 'Native prompt mode differs')
            # No cast of a cached vector: reject wrong native dtype at the boundary.
            require(prompt[0].ref_spk_embedding.dtype == torch.float32
                    and prompt[0].ref_code.dtype == torch.long, 'Native feature dtype differs')
            value = dict(native_ids(wrapper.processor,descriptor['text']),
                codes=prompt[0].ref_code.detach().cpu().clone(),
                speaker=prompt[0].ref_spk_embedding.detach().cpu().flatten().clone(),
                id=descriptor['id'], audio_sha256=descriptor['audio_sha256'],
                protocol_sha256=protocol_sha, descriptor_sha256=fingerprint(descriptor))
            tensors = validate_feature(value,descriptor,protocol_sha)
            require(all(v._version == version and v.grad is None for v,version in versions),
                    'Native extraction mutated state or accumulated gradients')
            pending = assets/('.pending-'+key); pending.mkdir()
            with (pending/'features.pt').open('xb') as stream:
                torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
            receipt = dict(id=descriptor['id'],protocol_sha256=protocol_sha,
                descriptor_sha256=fingerprint(descriptor),feature_sha256=digest(pending/'features.pt'),
                tensors=tensors,native_components=fingerprint(identity),fresh_native_extraction=True,
                original_unadapted_speaker=True,full_wav_samples=descriptor['wav_header']['samples'])
            write_new(pending/'receipt.json',receipt); pending.rename(directory)
        receipts[descriptor['id']] = digest(directory/'receipt.json')
        print(f'fresh native features: {number}/{len(p["records"])}',flush=True)
    require(native_component_identity(wrapper) == identity and all(v._version == version for v,version in versions),
            'Native components changed during extraction')
    verify(run)
    write_new(run/'extraction.json',dict(protocol_sha256=protocol_sha,records=len(receipts),
        asset_receipts=receipts,native_components=identity,all_native_state_unchanged=True,
        cached_features_used=False,all_features_fresh_cpu_fp32=True))


def verify_complete(run):
    run = external(run); p = verify(run); audit = read(run/'verification.json')
    require(audit['verified'] is True and audit['records'] == 324
            and audit['protocol_sha256'] == digest(run/'protocol.json')
            and audit['extraction_sha256'] == digest(run/'extraction.json'), 'Complete feature audit missing')
    require(audit['native_ids_verified'] == 324 and audit['all_tensor_contracts_verified'] is True
            and audit['all_sources_verified'] is True, 'Incomplete feature audit coverage')
    extraction = read(run/'extraction.json'); expected = {d['id'] for d in p['records']}
    require(set(audit['feature_sha256']) == expected == set(extraction['asset_receipts'])
            and extraction['protocol_sha256'] == audit['protocol_sha256']
            and extraction['records'] == 324 and extraction['all_native_state_unchanged'] is True
            and extraction['cached_features_used'] is False
            and extraction['all_features_fresh_cpu_fp32'] is True, 'Complete extraction bindings differ')
    return p


def audit(run):
    import torch
    run = external(run); p = verify(run); configure_numeric()
    require(not (run/'verification.json').exists(), 'Audit already complete')
    extraction = read(run/'extraction.json'); protocol_sha = digest(run/'protocol.json')
    require(extraction['protocol_sha256'] == protocol_sha and extraction['records'] == 324
            and extraction['all_native_state_unchanged'] is True
            and extraction['cached_features_used'] is False
            and extraction['all_features_fresh_cpu_fp32'] is True, 'Incomplete fresh extraction')
    expected = {d['id'] for d in p['records']}
    require(set(extraction['asset_receipts']) == expected
            and {x.name for x in (run/'assets').iterdir()} == {asset_key(i) for i in expected}, 'Wrong asset inventory')
    processor = load_native_processor(p['base'])
    features = {}
    for descriptor in p['records']:
        value,receipt = read_asset(run,descriptor,protocol_sha)
        require(digest(run/'assets'/asset_key(descriptor['id'])/'receipt.json')
                == extraction['asset_receipts'][descriptor['id']], 'Extraction receipt changed')
        require(receipt['native_components'] == fingerprint(extraction['native_components'])
                and receipt['fresh_native_extraction'] is True and receipt['original_unadapted_speaker'] is True
                and receipt['full_wav_samples'] == descriptor['wav_header']['samples'], 'Native provenance differs')
        tokens = native_ids(processor,descriptor['text'])
        require(all(torch.equal(value[k],tokens[k]) for k in tokens), 'Native role token IDs differ')
        features[descriptor['id']] = receipt['feature_sha256']
    write_new(run/'verification.json',dict(verified=True,records=324,protocol_sha256=protocol_sha,
        extraction_sha256=digest(run/'extraction.json'),feature_sha256=features,native_ids_verified=324,
        all_tensor_contracts_verified=True,all_sources_verified=True,
        native_generation_replay_parity_verified=False,training_updates=0,goal_achieved=False))


def load_feature(run, utterance_id):
    p = verify_complete(run)
    descriptor = next((d for d in p['records'] if d['id'] == utterance_id),None)
    require(descriptor is not None, 'Unplanned utterance requested')
    value,receipt = read_asset(run,descriptor,digest(Path(run)/'protocol.json'))
    require(receipt['feature_sha256'] == read(Path(run)/'verification.json')['feature_sha256'][utterance_id]
            and digest(Path(run)/'assets'/asset_key(utterance_id)/'receipt.json')
            == read(Path(run)/'extraction.json')['asset_receipts'][utterance_id],
            'Audited feature binding differs')
    return value


def load_features(run):
    """Verify once, then materialize the small complete feature set for training."""
    p = verify_complete(run); run = Path(run); audit = read(run/'verification.json')
    protocol_sha = digest(run/'protocol.json'); result = {}
    require(set(audit['feature_sha256']) == {d['id'] for d in p['records']}, 'Audit feature coverage differs')
    extraction = read(run/'extraction.json')
    for descriptor in p['records']:
        value,receipt = read_asset(run,descriptor,protocol_sha)
        require(receipt['feature_sha256'] == audit['feature_sha256'][descriptor['id']]
                and digest(run/'assets'/asset_key(descriptor['id'])/'receipt.json')
                == extraction['asset_receipts'][descriptor['id']], 'Audited feature/receipt changed')
        result[descriptor['id']] = value
    return p,result


def assert_native_components(wrapper, run):
    verify_complete(run)
    require(native_component_identity(wrapper) == read(Path(run)/'extraction.json')['native_components'],
            'Policy changed original codec or speaker encoder; fresh features cannot be reused')


def main():
    parser = argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','extract','audit'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--plan',type=Path)
    parser.add_argument('--driver',type=Path);parser.add_argument('--test-log',type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        require(all((args.plan,args.driver,args.test_log)), 'Preparation requires plan/driver/test-log')
        prepare(args.plan,args.run,args.driver,args.test_log)
    else:
        {'extract':extract,'audit':audit}[args.phase](args.run)


if __name__ == '__main__':
    main()
