package tensor

import "fmt"

func Fused1381SiluSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Log()
	return r, nil
}
func Fused1381SiluSubLogName() string { return "silu_sub_log" }
