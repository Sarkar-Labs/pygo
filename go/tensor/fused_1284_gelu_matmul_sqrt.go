package tensor

import "fmt"

func Fused1284GeluMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Sqrt()
	return r, nil
}
func Fused1284GeluMatmulSqrtName() string { return "gelu_matmul_sqrt" }
