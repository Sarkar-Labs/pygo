package tensor

import "fmt"

func Fused1380SiluAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Log()
	return r, nil
}
func Fused1380SiluAddLogName() string { return "silu_add_log" }
