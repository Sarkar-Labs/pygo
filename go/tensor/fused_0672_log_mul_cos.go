package tensor

import "fmt"

func Fused0672LogMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Cos()
	return r, nil
}
func Fused0672LogMulCosName() string { return "log_mul_cos" }
