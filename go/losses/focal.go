package losses

import "github.com/pygo/go/tensor"

func BinaryFocalLoss(logits, targets *tensor.Tensor, gamma, alpha float64) *tensor.Tensor {
	probs := logits.Sigmoid()
	bce := probs.Log().Neg().Mul(targets).Add(
		probs.MulScalar(-1).AddScalar(1).Log().Neg().Mul(targets.MulScalar(-1).AddScalar(1)))
	pt := targets.Mul(probs).Add(targets.MulScalar(-1).AddScalar(1).Mul(probs.MulScalar(-1).AddScalar(1)))
	weight := pt.MulScalar(-1).AddScalar(1).Pow(gamma)
	_ = alpha
	return bce.Mul(weight).Mean(-1, false)
}

func IoULoss(pred, target *tensor.Tensor) *tensor.Tensor {
	inter := pred.Mul(target).Sum(-1, false)
	union := pred.Sum(-1, false).Add(target.Sum(-1, false)).Sub(inter)
	return inter.AddScalar(1e-6).Div(union.AddScalar(1e-6)).MulScalar(-1).AddScalar(1)
}
