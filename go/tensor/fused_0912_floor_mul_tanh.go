package tensor

import "fmt"

func Fused0912FloorMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Tanh()
	return r, nil
}
func Fused0912FloorMulTanhName() string { return "floor_mul_tanh" }
