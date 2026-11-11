package tensor

import "fmt"

func Fused0589ExpMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Floor()
	return r, nil
}
func Fused0589ExpMatmulFloorName() string { return "exp_matmul_floor" }
