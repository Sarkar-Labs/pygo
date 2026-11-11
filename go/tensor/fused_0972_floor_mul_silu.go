package tensor

import "fmt"

func Fused0972FloorMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.SiLU()
	return r, nil
}
func Fused0972FloorMulSiluName() string { return "floor_mul_silu" }
