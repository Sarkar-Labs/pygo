package tensor

import "fmt"

func Fused1463NegDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Sqrt()
	return r, nil
}
func Fused1463NegDivSqrtName() string { return "neg_div_sqrt" }
