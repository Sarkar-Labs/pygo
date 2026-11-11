package tensor

import "fmt"

func Fused0914FloorMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Tanh()
	return r, nil
}
func Fused0914FloorMatmulTanhName() string { return "floor_matmul_tanh" }
