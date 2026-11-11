package tensor

import "fmt"

func Fused1688Log10DivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.GELU()
	return r, nil
}
func Fused1688Log10DivGeluName() string { return "log10_div_gelu" }
