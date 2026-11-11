package tensor

import "fmt"

func Fused1269GeluMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1269GeluMatmulSigmoidName() string { return "gelu_matmul_sigmoid" }
