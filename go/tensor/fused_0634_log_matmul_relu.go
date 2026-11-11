package tensor

import "fmt"

func Fused0634LogMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.ReLU()
	return r, nil
}
func Fused0634LogMatmulReluName() string { return "log_matmul_relu" }
