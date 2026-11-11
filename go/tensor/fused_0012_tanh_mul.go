package tensor

import "fmt"

func Fused0012TanhMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	return r, nil
}
func Fused0012TanhMulName() string { return "tanh_mul" }
