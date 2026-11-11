package tensor

import "fmt"

func Fused0939FloorMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Sin()
	return r, nil
}
func Fused0939FloorMatmulSinName() string { return "floor_matmul_sin" }
