package tensor

import "fmt"

func Fused1303GeluDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Cos()
	return r, nil
}
func Fused1303GeluDivCosName() string { return "gelu_div_cos" }
