package tensor

import "fmt"

func Fused0409AbsMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Floor()
	return r, nil
}
func Fused0409AbsMatmulFloorName() string { return "abs_matmul_floor" }
