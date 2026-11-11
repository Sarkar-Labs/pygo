package tensor

import "fmt"

func Fused0979FloorMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Neg()
	return r, nil
}
func Fused0979FloorMatmulNegName() string { return "floor_matmul_neg" }
