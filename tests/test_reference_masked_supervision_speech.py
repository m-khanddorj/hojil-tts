"""Artifact-binding tests; synthetic bytes only, no model or audio inference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import training
from training.engine import digest
from training.evaluate import ASR_REV
from training.reference_masked_supervision_evaluation import calibration_cases, validate_records
from training.reference_masked_supervision_plan import build_plan
from training.reference_masked_supervision_speech import bound_generation, verified_receipt
from scripts.score_latest_audio import score
from tests.test_reference_masked_supervision_plan import rows


class SpeechBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name); self.dest=self.run/'eval-000000'; self.dest.mkdir()
        for name in ('protocol.json','speech-protocol.json','native-verification.json','checkpoint-000000.pt'):
            (self.run/name).write_bytes(b'bound fixture')
        ref=self.run/'ref.wav'; ref.write_bytes(b'reference bytes')
        custom=self.run/'custom.wav'; custom.write_bytes(b'custom reference bytes')
        fit,cal=rows(47,'fit'),rows(16,'cal')
        for r in cal: r.update(ref_audio=str(ref),ref_audio_sha256=digest(ref),ref_text='reference text')
        cases=calibration_cases(build_plan(fit,cal),cal,dict(reference=str(custom),reference_sha256=digest(custom),
            reference_text='custom reference text',speaker_id='custom'))
        self.e=dict(cases=cases,decoding={'do_sample':False},features_audit_sha256='a'*64)
        self.generated=[]; self.scored=[]
        cp=digest(self.run/'checkpoint-000000.pt'); sp=digest(self.run/'speech-protocol.json')
        for i,c in enumerate(cases):
            wav=self.dest/f'{i}.wav'; wav.write_bytes(f'audio{i}'.encode())
            trajectory=self.dest/f'{i}.pt'; trajectory.write_bytes(f'codes{i}'.encode())
            r=dict(c,wav=wav.name,audio_sha256=digest(wav),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
                terminated=True,cosine=.985,step=0,checkpoint_sha256=cp,speech_protocol_sha256=sp)
            self.generated.append(r)
            self.scored.append(dict(r,hypothesis=c['text'],score=score(c['text'],c['text'],True),asr_revision=ASR_REV,
                decoding=self.e['decoding']))
        self.dump(self.dest/'generated.json',dict(records=self.generated,all_weights_unchanged=True,
            checkpoint_sha256=cp,speech_protocol_sha256=sp))
        summary=validate_records(cases,self.scored)
        self.dump(self.dest/'results.json',dict(records=self.scored,summary=summary,truncated=0))
        self.identity={'original_parameters_sha256':'b'*64,'half_weights_sha256':'c'*64}
        self.p={'frozen_identity':self.identity}
        self.dump(self.run/'teacher-000000.json',dict(step=0,protocol_sha256=digest(self.run/'protocol.json'),
            checkpoint_sha256=cp,frozen_identity=self.identity))
        self.receipt=dict(status='verified',step=0,recordings=80,summary=summary,truncated=0,
            checkpoint_sha256=cp,protocol_sha256=digest(self.run/'protocol.json'),speech_protocol_sha256=sp,
            results_sha256=digest(self.dest/'results.json'),generated_sha256=digest(self.dest/'generated.json'),
            native_verification_sha256=digest(self.run/'native-verification.json'),
            features_audit_sha256=self.e['features_audit_sha256'],native_inputs_verified=80,frozen_identity=self.identity,
            frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
            all_recomputed_cosines_finite=True,max_cosine_recomputation_error=0.,adapter_movement={'maximum':0.},goal_achieved=False,
            teacher_sha256=digest(self.run/'teacher-000000.json'),update_sha256={})
        self.dump(self.run/'verification-000000.json',self.receipt)
        stub=ModuleType('training.reference_masked_supervision_training')
        stub.checkpoint_path=lambda run,step:Path(run)/f'checkpoint-{step:06d}.pt'
        stub.read_checkpoint=lambda run,step:{'frozen_identity':self.identity,'adapter_movement':{'maximum':0.}}
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules,{'training.reference_masked_supervision_training':stub}).start()
        patch.object(training,'reference_masked_supervision_training',stub,create=True).start()

    @staticmethod
    def dump(path,value): path.write_text(json.dumps(value),encoding='utf-8')

    def test_complete_prior_evidence_verifies(self):
        self.assertEqual(verified_receipt(self.run,self.p,self.e,0),self.receipt)

    def test_changed_waveform_rejected_even_with_valid_receipt(self):
        (self.dest/self.generated[0]['wav']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Generation artifact'):
            verified_receipt(self.run,self.p,self.e,0)

    def test_scoring_cannot_change_any_generated_field(self):
        changed=copy.deepcopy(self.scored); changed[0]['cosine']=.99
        with self.assertRaisesRegex(ValueError,'Scoring changed'):
            bound_generation(self.run,self.e,0,changed)
        changed=copy.deepcopy(self.scored); changed[0]['unexpected']='value'
        with self.assertRaisesRegex(ValueError,'Scoring changed'):
            bound_generation(self.run,self.e,0,changed)

    def test_report_hash_and_recomputed_summary_both_required(self):
        report=json.loads((self.dest/'results.json').read_text()); report['truncated']=1
        self.dump(self.dest/'results.json',report)
        with self.assertRaisesRegex(ValueError,'provenance'):
            verified_receipt(self.run,self.p,self.e,0)
        self.receipt['results_sha256']=digest(self.dest/'results.json')
        self.dump(self.run/'verification-000000.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'score evidence'):
            verified_receipt(self.run,self.p,self.e,0)


if __name__=='__main__': unittest.main()
