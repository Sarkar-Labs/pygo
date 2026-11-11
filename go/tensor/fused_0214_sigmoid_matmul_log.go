package tensor

import "fmt"

func Fused0214SigmoidMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Log()
	return r, nil
}
func Fused0214SigmoidMatmulLogName() string { return "sigmoid_matmul_log" }
