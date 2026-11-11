package tensor

import "fmt"

func Fused0929FloorMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Exp()
	return r, nil
}
func Fused0929FloorMatmulExpName() string { return "floor_matmul_exp" }
