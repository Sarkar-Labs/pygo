package tensor

import "fmt"

func Fused0403AbsDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Cos()
	return r, nil
}
func Fused0403AbsDivCosName() string { return "abs_div_cos" }
