package tensor

import "fmt"

func Fused0682LogMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Ceil()
	return r, nil
}
func Fused0682LogMulCeilName() string { return "log_mul_ceil" }
