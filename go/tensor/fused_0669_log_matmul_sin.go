package tensor

import "fmt"

func Fused0669LogMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Sin()
	return r, nil
}
func Fused0669LogMatmulSinName() string { return "log_matmul_sin" }
