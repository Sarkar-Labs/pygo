package tensor

import "fmt"

func Fused1608Log2DivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Neg()
	return r, nil
}
func Fused1608Log2DivNegName() string { return "log2_div_neg" }
