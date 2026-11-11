package tensor

import "fmt"

func Fused0969FloorMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.GELU()
	return r, nil
}
func Fused0969FloorMatmulGeluName() string { return "floor_matmul_gelu" }
