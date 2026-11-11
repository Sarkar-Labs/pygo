package tensor

import "fmt"

func Fused0857CosMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Floor()
	return r, nil
}
func Fused0857CosMulFloorName() string { return "cos_mul_floor" }
