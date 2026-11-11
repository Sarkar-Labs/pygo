package tensor

import "fmt"

func Fused1290GeluAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Log()
	return r, nil
}
func Fused1290GeluAddLogName() string { return "gelu_add_log" }
