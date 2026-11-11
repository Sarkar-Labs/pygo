package tensor

import "fmt"

func Fused1103RoundDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Sqrt()
	return r, nil
}
func Fused1103RoundDivSqrtName() string { return "round_div_sqrt" }
