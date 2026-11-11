package tensor

import "fmt"

func Fused0267SigmoidMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Log10()
	return r, nil
}
func Fused0267SigmoidMulLog10Name() string { return "sigmoid_mul_log10" }
