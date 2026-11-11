package tensor

import "fmt"

func Fused0493SqrtDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Cos()
	return r, nil
}
func Fused0493SqrtDivCosName() string { return "sqrt_div_cos" }
