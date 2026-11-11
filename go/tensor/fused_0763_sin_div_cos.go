package tensor

import "fmt"

func Fused0763SinDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Cos()
	return r, nil
}
func Fused0763SinDivCosName() string { return "sin_div_cos" }
