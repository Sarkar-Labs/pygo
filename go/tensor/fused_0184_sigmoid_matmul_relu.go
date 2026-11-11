package tensor

import "fmt"

func Fused0184SigmoidMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.ReLU()
	return r, nil
}
func Fused0184SigmoidMatmulReluName() string { return "sigmoid_matmul_relu" }
