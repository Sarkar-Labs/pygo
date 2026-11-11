package tensor

import "fmt"

func Fused0703LogDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.SiLU()
	return r, nil
}
func Fused0703LogDivSiluName() string { return "log_div_silu" }
