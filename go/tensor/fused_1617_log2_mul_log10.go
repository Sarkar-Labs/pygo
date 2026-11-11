package tensor

import "fmt"

func Fused1617Log2MulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Log10()
	return r, nil
}
func Fused1617Log2MulLog10Name() string { return "log2_mul_log10" }
