package tensor

import "fmt"

func Fused0124ReluMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Log()
	return r, nil
}
func Fused0124ReluMatmulLogName() string { return "relu_matmul_log" }
