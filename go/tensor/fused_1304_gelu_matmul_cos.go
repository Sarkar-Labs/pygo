package tensor

import "fmt"

func Fused1304GeluMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Cos()
	return r, nil
}
func Fused1304GeluMatmulCosName() string { return "gelu_matmul_cos" }
