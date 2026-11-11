package tensor

import "fmt"

func Fused1489NegMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Floor()
	return r, nil
}
func Fused1489NegMatmulFloorName() string { return "neg_matmul_floor" }
