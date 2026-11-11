package tensor

import "fmt"

func Fused1264GeluMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.ReLU()
	return r, nil
}
func Fused1264GeluMatmulReluName() string { return "gelu_matmul_relu" }
