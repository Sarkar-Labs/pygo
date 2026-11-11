package tensor

import "fmt"

func Fused0570ExpAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Log()
	return r, nil
}
func Fused0570ExpAddLogName() string { return "exp_add_log" }
