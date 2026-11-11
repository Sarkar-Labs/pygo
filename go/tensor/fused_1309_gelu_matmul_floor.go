package tensor

import "fmt"

func Fused1309GeluMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Floor()
	return r, nil
}
func Fused1309GeluMatmulFloorName() string { return "gelu_matmul_floor" }
