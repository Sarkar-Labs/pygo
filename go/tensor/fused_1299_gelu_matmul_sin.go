package tensor

import "fmt"

func Fused1299GeluMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Sin()
	return r, nil
}
func Fused1299GeluMatmulSinName() string { return "gelu_matmul_sin" }
