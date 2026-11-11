package tensor

import "fmt"

func Fused0307TanhMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Sin()
	return r, nil
}
func Fused0307TanhMulSinName() string { return "tanh_mul_sin" }
