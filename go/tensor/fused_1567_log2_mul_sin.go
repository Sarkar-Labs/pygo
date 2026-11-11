package tensor

import "fmt"

func Fused1567Log2MulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Sin()
	return r, nil
}
func Fused1567Log2MulSinName() string { return "log2_mul_sin" }
