package tensor

import "fmt"

func Fused0058RoundDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	return r, nil
}
func Fused0058RoundDivName() string { return "round_div" }
