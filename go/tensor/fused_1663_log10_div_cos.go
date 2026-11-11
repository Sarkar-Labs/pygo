package tensor

import "fmt"

func Fused1663Log10DivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Cos()
	return r, nil
}
func Fused1663Log10DivCosName() string { return "log10_div_cos" }
