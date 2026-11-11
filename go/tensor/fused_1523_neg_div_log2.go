package tensor

import "fmt"

func Fused1523NegDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Log2()
	return r, nil
}
func Fused1523NegDivLog2Name() string { return "neg_div_log2" }
