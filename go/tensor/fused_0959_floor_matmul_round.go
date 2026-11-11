package tensor

import "fmt"

func Fused0959FloorMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Round()
	return r, nil
}
func Fused0959FloorMatmulRoundName() string { return "floor_matmul_round" }
