package losses

import "github.com/pygo/go/tensor"

func CrossEntropy(logits, targets *tensor.Tensor) *tensor.Tensor {
	return logits.Softmax(-1).Log().Neg().Mean(-1, false)
}
func LabelSmoothCE(logits, targets *tensor.Tensor, smooth float64) *tensor.Tensor {
	ce := CrossEntropy(logits, targets)
	entropy := logits.Softmax(-1).Log().Neg().Mean(-1, false)
	return ce.MulScalar(1 - smooth).Add(entropy.MulScalar(smooth))
}
func FocalLoss(logits, targets *tensor.Tensor, gamma, alpha float64) *tensor.Tensor {
	probs := logits.Sigmoid()
	bce := probs.Log().Neg()
	weight := probs.MulScalar(-1).AddScalar(1).Pow(gamma)
	_ = alpha
	return bce.Mul(weight).Mean(-1, false)
}
func DiceLoss(pred, target *tensor.Tensor, smooth float64) *tensor.Tensor {
	inter := pred.Mul(target).Sum(-1, false).Sum(-1, false)
	union := pred.Sum(-1, false).Sum(-1, false).Add(target.Sum(-1, false).Sum(-1, false))
	dice := inter.MulScalar(2).AddScalar(smooth).Div(union.AddScalar(smooth))
	return dice.MulScalar(-1).AddScalar(1)
}
