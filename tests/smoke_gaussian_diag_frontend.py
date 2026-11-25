"""Opt-in one-GPU smoke test; does not write checkpoints or train on VoxCeleb."""

import argparse
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from wespeaker.frontend.w2vbert import W2VBertFrontend
from wespeaker.models.gaussian_diag import GaussianDiag
from wespeaker.models.projections import get_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the opt-in frontend smoke test')
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    frontend = W2VBertFrontend(
        model_name=args.model, frozen=False, use_lora=False, layerdrop=0.0,
        gradient_checkpointing=True, max_input_samples=160000)
    model = GaussianDiag().cuda()
    model.add_module('frontend', frontend.cuda())
    wavs = torch.randn(2, 16000, device='cuda')
    lengths = torch.full((2,), 16000, device='cuda', dtype=torch.long)
    labels = torch.tensor([0, 1], device='cuda')
    with tempfile.TemporaryDirectory(prefix='wespeaker_diag_') as directory:
        dist.init_process_group('nccl', rank=0, world_size=1,
                                init_method=Path(directory, 'init').as_uri())
        try:
            for project_type, covariance in (
                    ('arc_margin_intertopk_subcenter_uncertainty', True),
                    ('arc_margin_intertopk_subcenter_gap_margin_detached_uncertainty', True)):
                projection = get_projection(dict(
                    project_type=project_type, embed_dim=256, num_class=8,
                    scale=32.0, easy_margin=False)).cuda()
                projection.update(0.2)
                model.add_module('projection', projection)
                ddp = torch.nn.parallel.DistributedDataParallel(model)
                ddp.train()
                for _ in range(2):
                    ddp.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        features, _ = ddp.module.frontend(wavs, lengths)
                        _, variance, embedding = ddp(features)
                        scores = (projection(embedding, variance, labels)
                                  if covariance else projection(embedding, labels))
                        loss = F.cross_entropy(scores, labels)
                    loss.backward()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite smoke-test loss')
                    missing = [name for name, param in model.named_parameters()
                               if param.requires_grad and param.grad is None]
                    if missing:
                        raise RuntimeError(f'Missing gradients: {missing}')
                    print(project_type, 'loss', loss.item(), 'variance',
                          tuple(variance.shape), 'layer_count', len(features))
                del ddp
                model.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                features, _ = model.frontend(wavs, lengths)
                _, variance, embedding = model(features)
            assert torch.isfinite(embedding).all() and torch.isfinite(variance).all()
            assert (variance > 0).all()
            assert (variance.std(dim=-1) > 0).all()
            print('Frontend + GaussianDiag + two classification heads + DDP backward: OK')
            print('Peak GPU memory MiB:', torch.cuda.max_memory_allocated() // 2**20)
        finally:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
