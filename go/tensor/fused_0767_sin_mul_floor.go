package tensor

import "fmt"

func Fused0767SinMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Floor()
	return r, nil
}
func Fused0767SinMulFloorName() string { return "sin_mul_floor" }
