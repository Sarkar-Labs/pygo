package tensor

import "fmt"

func Fused0701LogSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.SiLU()
	return r, nil
}
func Fused0701LogSubSiluName() string { return "log_sub_silu" }
