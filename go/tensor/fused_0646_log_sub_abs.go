package tensor

import "fmt"

func Fused0646LogSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Abs()
	return r, nil
}
func Fused0646LogSubAbsName() string { return "log_sub_abs" }
