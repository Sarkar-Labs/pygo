package tensor

import "fmt"

func Fused1329GeluMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.GELU()
	return r, nil
}
func Fused1329GeluMatmulGeluName() string { return "gelu_matmul_gelu" }
