package tensor

import "fmt"

func Fused1643Log10DivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Sqrt()
	return r, nil
}
func Fused1643Log10DivSqrtName() string { return "log10_div_sqrt" }
