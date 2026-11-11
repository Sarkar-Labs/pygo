package tensor

import "fmt"

func Fused1393SiluDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Cos()
	return r, nil
}
func Fused1393SiluDivCosName() string { return "silu_div_cos" }
