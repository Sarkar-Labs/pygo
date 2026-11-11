package tensor

import "fmt"

func Fused0537SqrtMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Log10()
	return r, nil
}
func Fused0537SqrtMulLog10Name() string { return "sqrt_mul_log10" }
