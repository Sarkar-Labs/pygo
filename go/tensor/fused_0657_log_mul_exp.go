package tensor

import "fmt"

func Fused0657LogMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Exp()
	return r, nil
}
func Fused0657LogMulExpName() string { return "log_mul_exp" }
