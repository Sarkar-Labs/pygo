package tensor

import "fmt"

func Fused1123RoundDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Cos()
	return r, nil
}
func Fused1123RoundDivCosName() string { return "round_div_cos" }
