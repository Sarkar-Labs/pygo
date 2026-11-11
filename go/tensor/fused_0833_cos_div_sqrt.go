package tensor

import "fmt"

func Fused0833CosDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Sqrt()
	return r, nil
}
func Fused0833CosDivSqrtName() string { return "cos_div_sqrt" }
