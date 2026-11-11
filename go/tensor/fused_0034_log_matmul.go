package tensor

import "fmt"

func Fused0034LogMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	return r, nil
}
func Fused0034LogMatmulName() string { return "log_matmul" }
