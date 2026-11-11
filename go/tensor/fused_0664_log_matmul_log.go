package tensor

import "fmt"

func Fused0664LogMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Log()
	return r, nil
}
func Fused0664LogMatmulLogName() string { return "log_matmul_log" }
