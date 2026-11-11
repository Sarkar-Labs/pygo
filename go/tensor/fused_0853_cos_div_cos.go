package tensor

import "fmt"

func Fused0853CosDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Cos()
	return r, nil
}
func Fused0853CosDivCosName() string { return "cos_div_cos" }
