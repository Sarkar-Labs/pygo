package tensor

import "fmt"

func Fused1527NegMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Log10()
	return r, nil
}
func Fused1527NegMulLog10Name() string { return "neg_mul_log10" }
