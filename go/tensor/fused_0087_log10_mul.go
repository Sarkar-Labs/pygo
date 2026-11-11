package tensor

import "fmt"

func Fused0087Log10Mul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	return r, nil
}
func Fused0087Log10MulName() string { return "log10_mul" }
