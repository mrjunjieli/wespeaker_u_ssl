import copy
import unittest

import torch
import torch.nn.functional as F

from wespeaker.models.gaussian_diag import GaussianDiag
from wespeaker.models.projections import get_projection
from wespeaker.utils.schedulers import ExponentialDecrease


class GaussianDiagTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(716)
        self.model = GaussianDiag(feat_dim=8, layer_num=3, head_num=4,
                                  compression_dim=5, embed_dim=6,
                                  pooling_head_chunk_size=2)
        self.x = torch.randn(2, 7, 3, 8)

    def test_diagonal_posterior(self):
        model = self.model.eval()
        key = torch.randn(2, 7, 5)
        value = torch.randn(2, 7, 5)
        mean, variance = model._pool_from_key(key, value)
        precision = F.linear(key, model.att_head.weight,
                             model.att_head.bias).reshape(2, 7, 4, 5).clamp(-15, 15).exp()
        prior = model.prior_logprec.clamp(-15, 15).exp()
        total = precision.sum(dim=1) + prior
        expected_mean = ((precision * value.unsqueeze(2)).sum(dim=1)
                         + prior * model.prior_mean) / total
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(variance, total.reciprocal())
        self.assertTrue(torch.all(variance.std(dim=-1) > 0))

    def test_tuple_and_tensor(self):
        model = self.model.eval()
        tensor_result = model(self.x)
        tuple_result = model(tuple(self.x.unbind(dim=2)))
        for tensor, sequence in zip(tensor_result, tuple_result):
            torch.testing.assert_close(tensor, sequence)
        self.assertEqual(tensor_result[1].shape, (2, 6))
        self.assertTrue(torch.all(tensor_result[1] > 0))

    def test_checkpoint_gradients(self):
        model = self.model.train()
        reference = copy.deepcopy(model)
        reference.pooling_checkpoint = False
        for network in (model, reference):
            _, variance, embedding = network(self.x)
            (embedding.square().mean() + variance.mean()).backward()
        for (name, param), (_, other) in zip(model.named_parameters(),
                                             reference.named_parameters()):
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            torch.testing.assert_close(param.grad, other.grad)

    def test_three_stage_heads(self):
        for project_type, covariance in (
                ('arc_margin_intertopk_subcenter_uncertainty', True),
                ('arc_margin_intertopk_subcenter_gap_margin_detached_uncertainty', True)):
            model = copy.deepcopy(self.model).train()
            head = get_projection(dict(project_type=project_type, embed_dim=6,
                                       num_class=8, scale=32.0, easy_margin=False))
            model.add_module('projection', head)
            head.update(0.2)
            _, variance, embedding = model(self.x)
            labels = torch.tensor([0, 1])
            scores = head(embedding, variance, labels) if covariance else head(embedding, labels)
            F.cross_entropy(scores, labels).backward()
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_gap_uncertainty_scale_stop_gradient(self):
        conf = dict(embed_dim=6, num_class=8, scale=32.0,
                    easy_margin=False, margin_span=0.0)
        head = get_projection(dict(
            conf, project_type='arc_margin_intertopk_subcenter_gap_margin_detached_uncertainty'))
        uncertainty_head = get_projection(dict(
            conf, project_type='arc_margin_intertopk_subcenter_uncertainty'))
        uncertainty_head.load_state_dict(head.state_dict())
        head.update(0.3)
        uncertainty_head.update(0.3)
        embedding = torch.randn(3, 6, requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        with torch.no_grad():
            head.weight.view(8, head.K, 6)[labels, 0] = (
                embedding + 0.5 * torch.randn_like(embedding))
        uncertainty_head.load_state_dict(head.state_dict())
        covariance = torch.full((3, 6), 0.5, requires_grad=True)
        scores = head(embedding, covariance, labels)
        torch.testing.assert_close(
            scores, uncertainty_head(embedding, covariance, labels))

        with torch.no_grad():
            cosine = F.linear(F.normalize(embedding), F.normalize(head.weight))
            cosine = cosine.reshape(3, 8, head.K).max(dim=2).values
            cosine = cosine.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
            mask = F.one_hot(labels, num_classes=8).bool()
            gap = cosine.gather(1, labels[:, None]) - cosine.masked_fill(
                mask, float('-inf')).max(dim=1, keepdim=True).values
            denominator = (embedding.square() * (
                covariance - gap + 2.0 + 1.0e-6).clamp_min(1.0e-6)).sum(
                    dim=1, keepdim=True).clamp_min(1.0e-6)
            multiplier = (torch.sqrt(
                embedding.square().sum(dim=1, keepdim=True) / denominator)
                * torch.exp(gap * torch.exp(
                    cosine.max(dim=1, keepdim=True).values))).clamp(0.2, 6)
        baseline = get_projection(dict(
            conf, project_type='arc_margin_intertopk_subcenter'))
        baseline.load_state_dict(head.state_dict())
        baseline.update(0.3)
        baseline_input = embedding.detach().clone().requires_grad_()
        reference_scores = baseline(baseline_input, labels) * multiplier
        torch.testing.assert_close(scores, reference_scores)
        F.cross_entropy(scores, labels).backward()
        F.cross_entropy(reference_scores, labels).backward()
        self.assertIsNone(covariance.grad)
        self.assertTrue(torch.isfinite(embedding.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        torch.testing.assert_close(embedding.grad, baseline_input.grad)
        torch.testing.assert_close(head.weight.grad, baseline.weight.grad)
        with torch.no_grad():
            changed_scores = head(embedding, covariance + 10.0, labels)
        self.assertFalse(torch.allclose(scores.detach(), changed_scores))

    def test_learning_rate_ratio(self):
        optimizer = torch.optim.AdamW([
            dict(params=[torch.nn.Parameter(torch.ones(1))]),
            dict(params=[torch.nn.Parameter(torch.ones(1))], lr_scale=0.1)])
        scheduler = ExponentialDecrease(optimizer, num_epochs=3, epoch_iter=10,
                                         initial_lr=1.0e-4, final_lr=1.0e-5,
                                         warm_up_epoch=0)
        for step in (0, 10, 29):
            scheduler.step(step)
            self.assertAlmostEqual(optimizer.param_groups[1]['lr'],
                                   optimizer.param_groups[0]['lr'] * 0.1)


if __name__ == '__main__':
    unittest.main()
