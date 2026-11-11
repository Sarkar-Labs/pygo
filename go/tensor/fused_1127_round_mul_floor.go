package tensor

import "fmt"

func Fused1127RoundMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Floor()
	return r, nil
}
func Fused1127RoundMulFloorName() string { return "round_mul_floor" }
