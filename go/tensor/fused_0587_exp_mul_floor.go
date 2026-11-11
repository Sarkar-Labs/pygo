package tensor

import "fmt"

func Fused0587ExpMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Floor()
	return r, nil
}
func Fused0587ExpMulFloorName() string { return "exp_mul_floor" }
