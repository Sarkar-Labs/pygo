package tensor

import "fmt"

func Fused0357TanhMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Log10()
	return r, nil
}
func Fused0357TanhMulLog10Name() string { return "tanh_mul_log10" }
