package tensor

import "fmt"

func Fused0300TanhAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Log()
	return r, nil
}
func Fused0300TanhAddLogName() string { return "tanh_add_log" }
