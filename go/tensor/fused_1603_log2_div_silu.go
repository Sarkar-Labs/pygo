package tensor

import "fmt"

func Fused1603Log2DivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.SiLU()
	return r, nil
}
func Fused1603Log2DivSiluName() string { return "log2_div_silu" }
