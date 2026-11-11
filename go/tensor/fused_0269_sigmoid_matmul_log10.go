package tensor

import "fmt"

func Fused0269SigmoidMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Log10()
	return r, nil
}
func Fused0269SigmoidMatmulLog10Name() string { return "sigmoid_matmul_log10" }
