"""GPU plumbing check only; no formal R/S training or scientific result."""
import copy
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Fresh directory for smoke-test artifacts")
    parser.add_argument("--model", help="Hugging Face model ID; default: the study's base pin "
                                        "(matched-dynamics/base_pin.json, or the file RSI_BASE_PIN names)")
    parser.add_argument("--revision", help="Immutable commit id of --model; required with --model")
    args = parser.parse_args()
    from rsi.base_pin import read_base_pin
    from rsi.shared_adapter import require_commit_id
    if (args.model is None) != (args.revision is None):
        parser.error("--model and --revision go together; the revision must be a commit id")
    if args.model is None:
        pin = read_base_pin()
        args.model, args.revision = pin["model"], pin["revision"]
    require_commit_id(args.revision, what="--revision")
    import torch
    from peft import LoraConfig, get_peft_model
    from rsi.backends import backend, response_losses
    from rsi.common import DEFAULTS, environment, rng_for, write_json, write_jsonl
    from rsi.tasks import make_instance, judge
    out = args.output
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit this smoke test to a GPU node')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'environment.json', dict(environment(), gpu=torch.cuda.get_device_name(0), PLUMBING_ONLY=True))
    # The whole suite under pytest, in its own interpreter: unittest discovery misses the pytest-style
    # tests (tests/test_gradient_recorder_fixes.py, tests/test_matching_fixes.py), and several tests
    # switch on deterministic kernels (rsi/determinism.py) that must not carry over into the GPU checks
    # below.  On a GPU node nothing may be skipped.
    report = out/'tests.xml'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests', '--junitxml', str(report)])
    assert report.is_file(), 'The test suite did not run; this check needs pytest (pip install pytest)'
    counts = {key: sum(int(suite.get(key, 0)) for suite in ElementTree.parse(report).getroot().iter('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
    assert counts['tests'] and not (counts['failures'] or counts['errors'] or counts['skipped']), \
        'Tests failed or skipped: %s (%s)' % (counts, report)
    torch.manual_seed(11)
    logits = torch.randn(3, 5, 7, device='cuda', requires_grad=True)
    labels = torch.tensor([[-100,-100,2,3,-100],[-100,1,4,-100,-100],[-100,-100,-100,2,5]],device='cuda')
    w = torch.tensor([.1,1.,.4], device='cuda')
    (response_losses(logits,labels)*w).sum().div(w.sum()).backward()
    expected = logits.grad.clone()
    logits.grad.zero_()
    for i in range(3):
        (response_losses(logits[i:i+1],labels[i:i+1])*w[i]/w.sum()).sum().backward()
    assert torch.allclose(expected,logits.grad,atol=1e-6,rtol=1e-4)
    assert logits.grad[:,-1].abs().sum().item()==0
    cfg=copy.deepcopy(DEFAULTS)
    cfg.update(model=args.model,revision=args.revision,device='cuda')
    cfg['generation'].update(batch_size=2,max_new_tokens=64)
    tasks=[make_instance(k,rng_for(12,k),'dev',0) for k in ('graph','arithmetic')]

    def prefill_logits(m):
        # The forward pass is bitwise reproducible across the reload.  Sampled decoding is not
        # guaranteed to be on GPU: Llama-3.2-1B on H100 flips near-tied tokens between two runs
        # of one model instance with one seed, so the sampled texts are recorded, not asserted.
        with torch.inference_mode():
            return m.model(**m.padded([m.prompt_ids(t['prompt']) for t in tasks],left=True)).logits.float().cpu()

    m=backend(cfg)
    try:
        torch.manual_seed(123)
        m.model=get_peft_model(m.model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj']))
        m.model.eval()
        before={n:p.detach().cpu().clone() for n,p in m.model.named_parameters() if p.requires_grad}
        m.model.save_pretrained(out/'adapter')
        first_logits=prefill_logits(m)
        first=m.generate(tasks,2,123)
    finally:
        m.close()
    m=backend(cfg,adapter=out/'adapter')
    try:
        loaded=dict(m.model.named_parameters())
        assert all(torch.equal(v,loaded[n].detach().cpu()) for n,v in before.items()), 'Adapter roundtrip mismatch'
        assert torch.equal(first_logits,prefill_logits(m)), 'Reload changed the forward pass'
        second=m.generate(tasks,2,123)
        same_text=[r['response'] for r in first]==[r['response'] for r in second]
        by_id={t['id']:t for t in tasks}
        rows=[dict(r,**judge(by_id[r['task_id']],r['response'])) for r in second]
        write_jsonl(out/'answers.jsonl',rows)
    finally:
        m.close()
    write_json(out/'complete.json',{'PLUMBING_ONLY':True,'cuda_loss_check':True,'adapter_roundtrip':True,
                                   'reload_logits_bitwise_equal':True,
                                   'fixed_seed_sampled_text_identical':same_text,'tasks':['graph','arithmetic'],
                                   'model':args.model,'revision':args.revision,
                                   'tests_run':counts['tests'],'scientific_training_performed':False})


if __name__ == "__main__":
    main()
