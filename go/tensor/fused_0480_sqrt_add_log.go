package tensor

import "fmt"

func Fused0480SqrtAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Log()
	return r, nil
}
func Fused0480SqrtAddLogName() string { return "sqrt_add_log" }
