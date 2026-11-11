package tensor

import "fmt"

func Fused0807SinMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Log10()
	return r, nil
}
func Fused0807SinMulLog10Name() string { return "sin_mul_log10" }
