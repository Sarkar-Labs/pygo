package tensor

import "fmt"

func Fused0219SigmoidMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Sin()
	return r, nil
}
func Fused0219SigmoidMatmulSinName() string { return "sigmoid_matmul_sin" }
