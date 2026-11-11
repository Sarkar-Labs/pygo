package tensor

import "fmt"

func Fused0924FloorMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Sqrt()
	return r, nil
}
func Fused0924FloorMatmulSqrtName() string { return "floor_matmul_sqrt" }
