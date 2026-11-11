package tensor

import "fmt"

func Fused0032LogMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	return r, nil
}
func Fused0032LogMulName() string { return "log_mul" }
