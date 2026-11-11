package tensor

import "fmt"

func Fused1618Log2DivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Log10()
	return r, nil
}
func Fused1618Log2DivLog10Name() string { return "log2_div_log10" }
