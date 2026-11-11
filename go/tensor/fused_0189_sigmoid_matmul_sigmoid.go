package tensor

import "fmt"

func Fused0189SigmoidMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Sigmoid()
	return r, nil
}
func Fused0189SigmoidMatmulSigmoidName() string { return "sigmoid_matmul_sigmoid" }
