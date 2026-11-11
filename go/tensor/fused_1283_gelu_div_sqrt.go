package tensor

import "fmt"

func Fused1283GeluDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Sqrt()
	return r, nil
}
func Fused1283GeluDivSqrtName() string { return "gelu_div_sqrt" }
