package tensor

import "fmt"

func Fused1307GeluMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Floor()
	return r, nil
}
func Fused1307GeluMulFloorName() string { return "gelu_mul_floor" }
