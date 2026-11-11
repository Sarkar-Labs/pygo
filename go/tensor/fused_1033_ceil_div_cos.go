package tensor

import "fmt"

func Fused1033CeilDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Cos()
	return r, nil
}
func Fused1033CeilDivCosName() string { return "ceil_div_cos" }
