package tensor

import "fmt"

func Fused0572ExpMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Log()
	return r, nil
}
func Fused0572ExpMulLogName() string { return "exp_mul_log" }
