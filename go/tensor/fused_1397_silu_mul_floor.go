package tensor

import "fmt"

func Fused1397SiluMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Floor()
	return r, nil
}
func Fused1397SiluMulFloorName() string { return "silu_mul_floor" }
