"""Meaningful numeric probes for real FP8 scaled GEMM; no robot I/O."""
import pytest
import torch
from fp8_runtime import FP8Linear

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('rows',[1,17,712])
def test_fp8_forward_shape_finite_and_bounded_random_error(rows):
    torch.manual_seed(123)
    weight=torch.randn(256,128,device='cuda',dtype=torch.bfloat16)*.02
    scale=(weight.abs().amax().float()/448).clamp_min(1e-12)
    module=FP8Linear((weight/scale).to(torch.float8_e4m3fn),scale)
    value=torch.randn(1,rows,128,device='cuda',dtype=torch.bfloat16)
    result=module(value)
    reference=torch.nn.functional.linear(value,weight)
    assert result.shape==(1,rows,256) and result.dtype==torch.bfloat16
    assert torch.isfinite(result).all()
    assert ((result.float()-reference.float()).norm()/reference.float().norm()).item()<.065
    assert torch.equal(module(torch.zeros_like(value)),torch.zeros_like(result))


def test_invalid_fp8_storage_and_scale_rejected():
    weight=torch.zeros(256,128,device='cuda')
    with pytest.raises(ValueError,match='E4M3'):
        FP8Linear(weight,torch.ones((),device='cuda'))
    with pytest.raises(ValueError,match='scale'):
        FP8Linear(weight.to(torch.float8_e4m3fn),torch.zeros((),device='cuda'))
