package tensor

import "fmt"

func Fused0645LogAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Abs()
	return r, nil
}
func Fused0645LogAddAbsName() string { return "log_add_abs" }
