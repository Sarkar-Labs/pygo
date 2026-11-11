package tensor

import "fmt"

func Fused0194SigmoidMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Tanh()
	return r, nil
}
func Fused0194SigmoidMatmulTanhName() string { return "sigmoid_matmul_tanh" }
