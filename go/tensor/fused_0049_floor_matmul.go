package tensor

import "fmt"

func Fused0049FloorMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	return r, nil
}
func Fused0049FloorMatmulName() string { return "floor_matmul" }
