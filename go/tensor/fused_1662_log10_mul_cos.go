package tensor

import "fmt"

func Fused1662Log10MulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Cos()
	return r, nil
}
func Fused1662Log10MulCosName() string { return "log10_mul_cos" }
