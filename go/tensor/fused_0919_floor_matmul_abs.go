package tensor

import "fmt"

func Fused0919FloorMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Abs()
	return r, nil
}
func Fused0919FloorMatmulAbsName() string { return "floor_matmul_abs" }
