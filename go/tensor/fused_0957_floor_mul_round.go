package tensor

import "fmt"

func Fused0957FloorMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Round()
	return r, nil
}
func Fused0957FloorMulRoundName() string { return "floor_mul_round" }
