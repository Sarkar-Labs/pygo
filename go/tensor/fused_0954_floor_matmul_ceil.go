package tensor

import "fmt"

func Fused0954FloorMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Ceil()
	return r, nil
}
func Fused0954FloorMatmulCeilName() string { return "floor_matmul_ceil" }
