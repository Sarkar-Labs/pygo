package tensor

import "fmt"

func Fused0069GeluMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	return r, nil
}
func Fused0069GeluMatmulName() string { return "gelu_matmul" }
