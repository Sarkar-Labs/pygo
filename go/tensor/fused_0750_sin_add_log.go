package tensor

import "fmt"

func Fused0750SinAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Log()
	return r, nil
}
func Fused0750SinAddLogName() string { return "sin_add_log" }
