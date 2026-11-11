package tensor

import "fmt"

func Fused0754SinMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Log()
	return r, nil
}
func Fused0754SinMatmulLogName() string { return "sin_matmul_log" }
