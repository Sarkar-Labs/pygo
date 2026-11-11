package tensor

import "fmt"

func Fused0627ExpMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Log10()
	return r, nil
}
func Fused0627ExpMulLog10Name() string { return "exp_mul_log10" }
