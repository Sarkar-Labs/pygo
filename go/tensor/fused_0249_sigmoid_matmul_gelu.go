package tensor

import "fmt"

func Fused0249SigmoidMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.GELU()
	return r, nil
}
func Fused0249SigmoidMatmulGeluName() string { return "sigmoid_matmul_gelu" }
