package tensor

import "fmt"

func Fused0639LogMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Sigmoid()
	return r, nil
}
func Fused0639LogMatmulSigmoidName() string { return "log_matmul_sigmoid" }
