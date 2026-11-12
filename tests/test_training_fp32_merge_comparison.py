import unittest
from training.fp32_merge_comparison import merge_fp32


class FP32MergeTests(unittest.TestCase):
    def fixture(self):
        import torch
        from torch import nn
        model=nn.Module();model.talker=nn.Module();model.talker.projection=nn.Linear(3,2)
        model.speaker_encoder=nn.Linear(3,2)
        with torch.no_grad():model.talker.projection.weight.fill_(1)
        state={'projection.lora_A.weight':torch.tensor([[.001,.002,.003]]),
               'projection.lora_B.weight':torch.tensor([[.01],[.02]])}
        return model,state

    def test_preserves_updates_that_bf16_export_rounds_away(self):
        import torch
        model,state=self.fixture();before={k:v.clone() for k,v in model.state_dict().items()}
        merge_fp32(model,state,.5)
        weight=model.talker.projection.weight
        self.assertTrue((weight>before['talker.projection.weight']).all())
        self.assertTrue(torch.equal(weight.bfloat16(),before['talker.projection.weight'].bfloat16()))
        x=torch.tensor([[.1,.2,.3]])
        expected=torch.nn.functional.linear(x,before['talker.projection.weight'],before['talker.projection.bias'])
        expected += .5*torch.nn.functional.linear(torch.nn.functional.linear(x,state['projection.lora_A.weight']),state['projection.lora_B.weight'])
        torch.testing.assert_close(model.talker.projection(x),expected,atol=1e-7,rtol=1e-6)
        for k,v in model.state_dict().items():
            if k!='talker.projection.weight':self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_zero_strength_is_exact_and_invalid_input_does_not_mutate(self):
        import torch
        model,state=self.fixture();before={k:v.clone() for k,v in model.state_dict().items()}
        merge_fp32(model,state,0)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        for strength in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):merge_fp32(model,state,strength)
        state['projection.lora_B.weight'][0,0]=float('nan')
        with self.assertRaises(ValueError):merge_fp32(model,state,.5)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def test_rejects_low_precision_and_incomplete_adapters(self):
        model,state=self.fixture();state['projection.lora_A.weight']=state['projection.lora_A.weight'].bfloat16()
        with self.assertRaises(ValueError):merge_fp32(model,state,.5)
        del state['projection.lora_B.weight']
        with self.assertRaises(ValueError):merge_fp32(model,state,.5)


if __name__=='__main__':unittest.main()
