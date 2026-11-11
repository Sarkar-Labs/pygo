package tensor

import "fmt"

func Fused1702Log10MulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Log2()
	return r, nil
}
func Fused1702Log10MulLog2Name() string { return "log10_mul_log2" }
