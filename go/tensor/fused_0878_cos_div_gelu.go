package tensor

import "fmt"

func Fused0878CosDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.GELU()
	return r, nil
}
func Fused0878CosDivGeluName() string { return "cos_div_gelu" }
