package tensor

import "fmt"

func Fused0647LogMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Abs()
	return r, nil
}
func Fused0647LogMulAbsName() string { return "log_mul_abs" }
