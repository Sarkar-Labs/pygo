package tensor

import "fmt"

func Fused1163RoundDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Log2()
	return r, nil
}
func Fused1163RoundDivLog2Name() string { return "round_div_log2" }
