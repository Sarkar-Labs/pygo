package tensor

import "fmt"

func Fused1347GeluMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Log10()
	return r, nil
}
func Fused1347GeluMulLog10Name() string { return "gelu_mul_log10" }
