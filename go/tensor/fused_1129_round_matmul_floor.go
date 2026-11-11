package tensor

import "fmt"

func Fused1129RoundMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Floor()
	return r, nil
}
func Fused1129RoundMatmulFloorName() string { return "round_matmul_floor" }
