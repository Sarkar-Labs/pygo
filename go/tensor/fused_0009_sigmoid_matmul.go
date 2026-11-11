package tensor

import "fmt"

func Fused0009SigmoidMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	return r, nil
}
func Fused0009SigmoidMatmulName() string { return "sigmoid_matmul" }
