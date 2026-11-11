package tensor

import "fmt"

func Fused0497SqrtMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Floor()
	return r, nil
}
func Fused0497SqrtMulFloorName() string { return "sqrt_mul_floor" }
