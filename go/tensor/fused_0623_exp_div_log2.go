package tensor

import "fmt"

func Fused0623ExpDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Log2()
	return r, nil
}
func Fused0623ExpDivLog2Name() string { return "exp_div_log2" }
